#client.py
# This file is going to contain all of our centralized machine learning workflow
# Centralized Machine learning: the participants/clients are connected with the central server to upload their data
import argparse
from flwr.client import start_client, start_numpy_client
import os
from flwr.client import NumPyClient
import tensorflow as tf
from flwr_datasets import FederatedDataset
import warnings
from flwr_datasets.partitioner import IidPartitioner
warnings.filterwarnings("ignore", category=UserWarning)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

# Parse arguments
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=list(range(50)),
    default=0,
    help="Partition of the dataset (0-49). "
    "The dataset is divided into 50 partitions created artificially.",
)
args, _ = parser.parse_known_args()

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Download and partition dataset
fds = FederatedDataset(dataset="cifar10", partitioners={"train": 50})
partition = fds.load_partition(args.partition_id, "train")
partition.set_format("numpy")

# Divide data on each node: 80% train, 20% test
partition = partition.train_test_split(test_size=0.2, seed=42)
x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]


# Define Flower client-here actually the federated learning magic happening
class FlowerClient(NumPyClient):
    def get_parameters(self, config):  #this function needs to exsist if on the server side we don't initialize any weight the server is actually going to pick a random client from those they are connected and call this function to initialize it's weight
        return model.get_weights()

    def fit(self, parameters, config): #parameters = the parameters are sent from server to the client for a given round, config = It is a dictionary of strings to any scholar, this is also passes from the server to the client
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {} #return 3 things weight of the model after traning, length of the traning data (if the clients have different sizes of data we might want on the server side to aggregate them differently), a empty dictionary
 
    def evaluate(self, parameters, config): # this parameter, server has aggregated all the parameters from the client after the training, it is going to send back the aggregated weight to every client for them to evaluate this new model on their data
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient()


# Legacy mode
if __name__ == "__main__":
    # starting the client with the server address
    # In the client side this address is going to be my local host
    start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(), grpc_max_message_length = 1024*1024*1024)