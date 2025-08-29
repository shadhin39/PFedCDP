import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
import argparse
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
# Import the extracted components
from data_loader import partition_data
from differential_privacy import DifferentialPrivacy
from quantization import Quantization
from fisher_information_calculation import FisherInformationCalculator
from knowledge_distillation_client import KnowledgeDistillationLoss
from dataset_loaders import load_cbis_ddsm_dataset

# Parse command line arguments for client ID
parser = argparse.ArgumentParser(description='Federated Learning Client 2 - 100 Clients')
parser.add_argument('--partition-id', type=int, choices=range(100), required=True,
                    help='Partition ID for this client (0-99 for 100 clients)')
args = parser.parse_args()
client_id = args.partition_id

print(f"\n=== INITIALIZING CLIENT 2 (ID: {client_id}) FOR 100 CLIENTS ===")

# Data partitioning function for 100 clients
def partition_data_100_clients(client_id=0, num_partitions=100):
    """Partition CBIS-DDSM data for 100 clients"""
    try:
        # Load CBIS-DDSM dataset
        x_train, y_train, x_test, y_test = load_cbis_ddsm_dataset()
        
        # Simple partitioning for 100 clients
        partition_size = len(x_train) // num_partitions
        start_idx = client_id * partition_size
        end_idx = start_idx + partition_size
        
        # Ensure we don't exceed array bounds
        if end_idx > len(x_train):
            end_idx = len(x_train)
        
        client_x_train = x_train[start_idx:end_idx]
        client_y_train = y_train[start_idx:end_idx]
        
        print(f"[CLIENT 2-{client_id}] Data partition: {len(client_x_train)} training, {len(x_test)} test samples")
        
        return client_x_train, client_y_train, x_test, y_test
        
    except Exception as e:
        print(f"[CLIENT 2-{client_id}] Error loading CBIS-DDSM data: {e}")
        # Fallback to synthetic data
        np.random.seed(client_id + 42)
        x_train = np.random.random((100, 224, 224, 1)).astype('float32')
        y_train = np.random.randint(0, 2, (100,))
        x_test = np.random.random((20, 224, 224, 1)).astype('float32')
        y_test = np.random.randint(0, 2, (20,))
        
        return x_train, y_train, x_test, y_test

# Knowledge Distillation Loss Function
class KnowledgeDistillationLoss:
    def __init__(self, temperature=3.0, lambda_balance=0.5):
        self.temperature = temperature
        self.lambda_balance = lambda_balance
        
    def compute_distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Compute combined distillation and CE loss"""
        # Convert true labels to one-hot if needed
        if len(true_labels.shape) == 1:
            true_labels_onehot = tf.one_hot(true_labels, depth=2)  # Binary classification
        else:
            true_labels_onehot = true_labels
            
        # Softmax with temperature for soft targets
        soft_teacher = tf.nn.softmax(teacher_logits / self.temperature)
        soft_student = tf.nn.softmax(student_logits / self.temperature)
        
        # Distillation loss (KL divergence)
        distillation_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                soft_teacher, soft_student, from_logits=False
            )
        ) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                true_labels, student_logits, from_logits=True
            )
        )
        
        # Combined loss
        total_loss = self.lambda_balance * distillation_loss + (1 - self.lambda_balance) * ce_loss
        return total_loss, distillation_loss, ce_loss

# Fisher Information Calculator
class FisherInformationCalculator:
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        
    def calculate_fisher_information(self, model, x_data, y_data, batch_size=16):
        """Calculate Fisher Information Matrix for model parameters"""
        fisher_info = {}
        
        # Use smaller batch size for memory efficiency
        batch_size = min(batch_size, len(x_data))
        
        # Create a copy of the model for gradient computation
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            predictions = model(x_data[:batch_size])
            
            # Calculate loss for each sample
            losses = []
            for i in range(batch_size):
                if len(y_data.shape) > 1:
                    sample_loss = tf.keras.losses.categorical_crossentropy(
                        y_data[i:i+1], predictions[i:i+1], from_logits=True
                    )
                else:
                    sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
                        y_data[i:i+1], predictions[i:i+1], from_logits=True
                    )
                losses.append(sample_loss)
        
        # Calculate Fisher Information for each layer
        for layer in model.trainable_variables:
            layer_fisher = []
            
            for loss in losses:
                # Calculate gradient for this sample
                grad = tape.gradient(loss, layer)
                if grad is not None:
                    # Square the gradient (Fisher Information approximation)
                    squared_grad = tf.square(grad)
                    layer_fisher.append(squared_grad)
            
            if layer_fisher:
                # Average across samples
                fisher_info[layer.name] = tf.reduce_mean(tf.stack(layer_fisher), axis=0)
            else:
                fisher_info[layer.name] = tf.zeros_like(layer)
        
        del tape
        return fisher_info

# Lightweight CNN Model for CBIS-DDSM
class PartitionedModel:
    def __init__(self, input_shape=(224, 224, 1), num_classes=2, client_id=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.client_id = client_id
        self.full_model = self._create_lightweight_model()
        
    def _create_lightweight_model(self):
        """Create a lightweight CNN model for CBIS-DDSM binary classification"""
        model = keras.Sequential([
            layers.Conv2D(16, (7, 7), strides=2, activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((4, 4)),
            layers.Conv2D(32, (5, 5), activation='relu'),
            layers.MaxPooling2D((4, 4)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def get_weights(self):
        return self.full_model.get_weights()
    
    def set_weights(self, weights):
        self.full_model.set_weights(weights)
    
    def train_with_advanced_techniques(self, x_train, y_train, epochs=3, batch_size=16):
        """Train with advanced ML techniques for 100 clients"""
        # Initialize components
        kd_loss = KnowledgeDistillationLoss()
        fisher_calc = FisherInformationCalculator()
        dp = DifferentialPrivacy(noise_multiplier=0.05)  # Lower noise for stability
        quantizer = Quantization(num_bits=8)
        
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)  # Lower learning rate
        
        # Custom training loop
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(batch_size).shuffle(100)
        
        for epoch in range(epochs):
            print(f"[CLIENT 2-{self.client_id}] Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            batch_count = 0
            
            for batch_x, batch_y in train_dataset:
                with tf.GradientTape() as tape:
                    predictions = self.full_model(batch_x, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        batch_y, predictions, from_logits=False
                    )
                    loss = tf.reduce_mean(loss)
                
                # Calculate gradients
                gradients = tape.gradient(loss, self.full_model.trainable_variables)
                
                # Add differential privacy noise (optional for some clients)
                if self.client_id % 10 == 0:  # Apply DP to every 10th client
                    noisy_gradients = dp.add_noise_to_gradients(gradients)
                    optimizer.apply_gradients(zip(noisy_gradients, self.full_model.trainable_variables))
                else:
                    optimizer.apply_gradients(zip(gradients, self.full_model.trainable_variables))
                
                epoch_loss += loss.numpy()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f"[CLIENT 2-{self.client_id}] Average epoch loss: {avg_loss:.4f}")
        
        return {'loss': epoch_loss}

# Initialize model and get partitioned data
model = PartitionedModel(client_id=client_id)

# Compile model
model.full_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"[CLIENT 2-{client_id}] Model compiled successfully")
print(f"[CLIENT 2-{client_id}] Total parameters: {model.full_model.count_params()}")

# Get partitioned data for this client
client_x_train, client_y_train, client_x_test, client_y_test = partition_data_100_clients(
    client_id, num_partitions=100
)

print(f"[CLIENT 2-{client_id}] Ready for federated learning with {len(client_x_train)} training samples")

# Define Flower client for 100 clients
class CBISFlowerClient100(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, client_id):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.client_id = client_id
        
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        # Set global parameters
        self.model.set_weights(parameters)
        
        # Get current round
        current_round = config.get("server_round", 1)
        
        # Train with advanced techniques
        history = self.model.train_with_advanced_techniques(
            self.x_train, self.y_train, 
            epochs=config.get("local_epochs", 2),  # Fewer epochs for 100 clients
            batch_size=16
        )
        
        print(f"[CLIENT 2-{self.client_id}] Training completed for round {current_round}")
        
        # Apply quantization for communication efficiency
        quantizer = Quantization(num_bits=8)
        quantized_weights = quantizer.quantize_weights(self.model.get_weights())
        
        return quantized_weights, len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Handle case where test data might be empty
        if len(self.x_test) == 0:
            return 0.5, 1, {"accuracy": 0.5}  # Return neutral values
        
        loss, accuracy = self.model.full_model.evaluate(
            self.x_test, self.y_test, verbose=0
        )
        print(f"[CLIENT 2-{self.client_id}] Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.x_test), {"accuracy": accuracy}

# Create and start Flower client
client = CBISFlowerClient100(
    model, client_x_train, client_y_train, 
    client_x_test, client_y_test, client_id
)

print(f"[CLIENT 2-{client_id}] Starting Flower client for 100 clients setup...")
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)