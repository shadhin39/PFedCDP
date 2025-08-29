# client_cbis_ddsm_100.py
# Federated Learning Client for CBIS-DDSM Dataset - 100 clients setup
import argparse
from flwr.client import start_client, start_numpy_client
import os
from flwr.client import NumPyClient
import tensorflow as tf
from tensorflow import keras
from keras import layers
import warnings
from dataset_loaders import load_cbis_ddsm_dataset
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

# Parse arguments
parser = argparse.ArgumentParser(description="Flower CBIS-DDSM Client - 100 clients")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=list(range(100)),
    default=0,
    help="Partition of the dataset (0-99). "
    "The dataset is divided into 100 partitions.",
)
args, _ = parser.parse_known_args()

# Create CNN model for CBIS-DDSM (224x224x1 input, 2 classes: Benign/Malignant)
def create_cbis_model():
    model = keras.Sequential([
        keras.Input(shape=(224, 224, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')  # 2 classes: Benign/Malignant
    ])
    return model

# Load and compile model
model = create_cbis_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load CBIS-DDSM dataset
print("Loading CBIS-DDSM dataset...")
x_train_full, y_train_full, x_test, y_test = load_cbis_ddsm_dataset()

# Partition data for federated learning
def partition_data(x_data, y_data, partition_id, num_partitions=100):
    """Partition data for federated learning"""
    n_samples = len(x_data)
    partition_size = n_samples // num_partitions
    start_idx = partition_id * partition_size
    
    if partition_id == num_partitions - 1:  # Last partition gets remaining data
        end_idx = n_samples
    else:
        end_idx = start_idx + partition_size
    
    return x_data[start_idx:end_idx], y_data[start_idx:end_idx]

# Get partition for this client
x_train, y_train = partition_data(x_train_full, y_train_full, args.partition_id, 100)

print(f"Client {args.partition_id} - Train: {x_train.shape}, Test: {x_test.shape}")
print(f"Classes distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

# Define Flower client for CBIS-DDSM
class CBISFlowerClient100(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=16, verbose=0)
        return model.get_weights(), len(x_train), {
            "accuracy": float(model.evaluate(x_train, y_train, verbose=0)[1])
        }

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return float(loss), len(x_test), {
            "accuracy": float(accuracy)
        }

def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return CBISFlowerClient100()

if __name__ == "__main__":
    print(f"Starting CBIS-DDSM client {args.partition_id} (100-client setup)...")
    start_numpy_client(
        server_address="127.0.0.1:8080",
        client=CBISFlowerClient100(),
        grpc_max_message_length=1024*1024*1024
    )