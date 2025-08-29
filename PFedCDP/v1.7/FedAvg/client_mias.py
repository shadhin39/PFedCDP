# client_mias.py
# Federated Learning Client for MIAS Dataset
import argparse
from flwr.client import start_client, start_numpy_client
import os
from flwr.client import NumPyClient
import tensorflow as tf
from tensorflow import keras
from keras import layers
import warnings
from dataset_loaders import load_mias_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

# Parse arguments
parser = argparse.ArgumentParser(description="Flower MIAS Client")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=[0, 1, 2],
    default=0,
    help="Partition of the dataset (0, 1 or 2). "
    "The dataset is divided into 3 partitions.",
)
args, _ = parser.parse_known_args()

# Create CNN model for MIAS (224x224x1 input, 3 classes: Normal/Benign/Malignant)
def create_mias_model():
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
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # 3 classes: Normal, Benign, Malignant
    ])
    return model

# Load and compile model
model = create_mias_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load MIAS dataset
print("Loading MIAS dataset...")
x_train_full, y_train_full, x_test, y_test = load_mias_dataset()

# Partition data for federated learning
def partition_data(x_data, y_data, partition_id, num_partitions=3):
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
x_train, y_train = partition_data(x_train_full, y_train_full, args.partition_id)

print(f"Client {args.partition_id} - Train: {x_train.shape}, Test: {x_test.shape}")
print(f"Classes distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

# Define Flower client for MIAS
class MIASFlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        
        # Train with early stopping
        history = model.fit(
            x_train, y_train,
            epochs=5,  # Reduced epochs for federated rounds
            batch_size=16,  # Smaller batch size for medical images
            validation_split=0.2,
            verbose=1
        )
        
        return model.get_weights(), len(x_train), {
            "train_accuracy": float(history.history['accuracy'][-1]),
            "train_loss": float(history.history['loss'][-1])
        }

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Additional metrics
        predictions = model.predict(x_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_pred_proba = predictions
        
        # Calculate precision, recall, and F1 score
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate AUC (for multi-class classification)
        try:
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc = 0.0
        
        # Calculate specificity and sensitivity
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:  # Multi-class classification
            specificity_scores = []
            sensitivity_scores = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp
                
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                specificity_scores.append(spec)
                sensitivity_scores.append(sens)
            
            specificity = np.mean(specificity_scores)
            sensitivity = np.mean(sensitivity_scores)
        
        return float(loss), len(x_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(auc),
            "specificity": float(specificity),
            "sensitivity": float(sensitivity),
            "test_samples": len(x_test)
        }

def client_fn(cid: str):
    """Create and return an instance of Flower Client for MIAS."""
    return MIASFlowerClient()

# Legacy mode
if __name__ == "__main__":
    print(f"Starting MIAS client {args.partition_id}...")
    start_numpy_client(
        server_address="127.0.0.1:8081",  # Different port for MIAS
        client=MIASFlowerClient(),
        grpc_max_message_length=1024*1024*1024
    )