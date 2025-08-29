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

# Load dataset
x_train, y_train, x_test, y_test = partition_data(client_id)

# Partitioned Model B with lighter architecture
class PartitionedModelB:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        # Feature Extractor (W_L,k) - shared with server (lighter architecture)
        self.feature_extractor = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu")
        ])
        
        # Client-specific Classifier (W_C,k) - NOT shared
        self.classifier = keras.Sequential([
            keras.Input(shape=(64,)),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])
        
        # Full model for training
        self.full_model = keras.Sequential([
            self.feature_extractor,
            self.classifier
        ])
    
    def get_feature_extractor_weights(self):
        return self.feature_extractor.get_weights()
    
    def set_feature_extractor_weights(self, weights):
        self.feature_extractor.set_weights(weights)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.partitioned_model = PartitionedModelB()
        self.partitioned_model.full_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
    
    def get_parameters(self, config):
        # Only return feature extractor weights
        return self.partitioned_model.get_feature_extractor_weights()
    
    def fit(self, parameters, config):
        # Set feature extractor weights from server
        self.partitioned_model.set_feature_extractor_weights(parameters)
        
        # Train the full model (feature extractor + classifier)
        self.partitioned_model.full_model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
        
        # Return only feature extractor weights
        return self.partitioned_model.get_feature_extractor_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # Set feature extractor weights from server
        self.partitioned_model.set_feature_extractor_weights(parameters)
        
        # Evaluate the full model
        loss, accuracy = self.partitioned_model.full_model.evaluate(x_test, y_test, verbose=1)
        return loss, len(x_test), {"accuracy": accuracy}

# Connect to server on port 8081
fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=FlowerClient())