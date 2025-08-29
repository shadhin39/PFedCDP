import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

def partition_data(client_id):
    # Partitioning the data using Dirichlet distribution to ensure non-IID
    partitioner = DirichletPartitioner(
        num_partitions=10, partition_by="label",
        alpha=0.5, min_partition_size=10, self_balancing=True
    )
    # Changed dataset to "fashion_mnist" to load FMNIST instead of MNIST
    fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": partitioner})

    partition = fds.load_partition(client_id, split="train")
    print(partition[client_id])
    partition_sizes = [
        len(fds.load_partition(partition_id)) for partition_id in range(10)
    ]
    print(sorted(partition_sizes))
    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train = [np.array(img).reshape(28, 28, 1) for img in partition["train"]["image"]]
    y_train = np.array(partition["train"]["label"])

    x_test = [np.array(img).reshape(28, 28, 1) for img in partition["test"]["image"]]
    y_test = np.array(partition["test"]["label"])

    # Convert to NumPy arrays and normalize
    x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0

    return x_train, y_train, x_test, y_test

client_id = int(input("Enter client ID for Model B (e.g., 1, 2, 3...): "))

# Define Model B
def create_model_B():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    return model

# Load dataset
x_train, y_train, x_test, y_test = partition_data(client_id)

# Initialize Model B
model = create_model_B()
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        return loss, len(x_test), {"accuracy": accuracy}

# Connect to server on port 8081
fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=FlowerClient())