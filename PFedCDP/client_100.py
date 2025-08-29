#client_100.py
# Federated Learning Client for 100 clients setup
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
parser = argparse.ArgumentParser(description="Flower 100 Clients")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=list(range(100)),
    default=0,
    help="Partition of the dataset (0-99). "
    "The dataset is divided into 100 partitions created artificially.",
)
args, _ = parser.parse_known_args()

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Download and partition dataset
fds = FederatedDataset(dataset="cifar10", partitioners={"train": 100})
partition = fds.load_partition(args.partition_id, "train")
partition.set_format("numpy")

# Divide data on each node: 80% train, 20% test
partition = partition.train_test_split(test_size=0.2, seed=42)
x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]


# Define Flower client for 100 clients setup
class FlowerClient100(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}
 
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient100()


# Legacy mode
if __name__ == "__main__":
    print(f"Starting 100-client setup - Client {args.partition_id}...")
    start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient100(), grpc_max_message_length = 1024*1024*1024)