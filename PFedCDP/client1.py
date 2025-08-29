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

# Parse command line arguments for client ID
parser = argparse.ArgumentParser(description='Federated Learning Client 1')
parser.add_argument('--partition-id', type=int, choices=range(50), required=True,
                    help='Partition ID for this client (0-49 for 50 clients)')
args = parser.parse_args()
client_id = args.partition_id

print(f"\n=== INITIALIZING CLIENT 1 (ID: {client_id}) ===")

# Updated partition_data function for 50 clients
def partition_data(dataset_name="fashion_mnist", client_id=0, num_partitions=50):
    """Partition data using Dirichlet distribution for 50 clients"""
    try:
        # Load FederatedDataset
        fds = FederatedDataset(dataset=dataset_name, partitioners={
            "train": DirichletPartitioner(num_partitions=num_partitions, partition_by="label", alpha=0.5)
        })
        
        # Get partition for this client
        partition = fds.load_partition(client_id, "train")
        
        # Convert to numpy arrays
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        
        x_train = np.array([x['image'] for x in partition_train_test['train']])
        y_train = np.array([x['label'] for x in partition_train_test['train']])
        x_test = np.array([x['image'] for x in partition_train_test['test']])
        y_test = np.array([x['label'] for x in partition_train_test['test']])
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        print(f"[CLIENT 1-{client_id}] Data partition: {len(x_train)} training, {len(x_test)} test samples")
        
        return x_train, y_train, x_test, y_test
        
    except Exception as e:
        print(f"[CLIENT 1-{client_id}] Error in data partitioning: {e}")
        # Fallback to Fashion-MNIST loading
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        # Simple partitioning as fallback
        partition_size = len(x_train) // num_partitions
        start_idx = client_id * partition_size
        end_idx = start_idx + partition_size
        
        x_train = x_train[start_idx:end_idx].astype('float32') / 255.0
        y_train = y_train[start_idx:end_idx]
        x_test = x_test.astype('float32') / 255.0
        
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
            true_labels_onehot = tf.one_hot(true_labels, depth=10)
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

# Differential Privacy Class
class DifferentialPrivacy:
    def __init__(self, noise_multiplier=0.1, l2_norm_clip=1.0):
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        
    def add_noise_to_gradients(self, gradients):
        """Add Gaussian noise to gradients for differential privacy"""
        noisy_gradients = []
        for grad in gradients:
            if grad is not None:
                # Clip gradients
                clipped_grad = tf.clip_by_norm(grad, self.l2_norm_clip)
                # Add Gaussian noise
                noise = tf.random.normal(tf.shape(clipped_grad), stddev=self.noise_multiplier)
                noisy_grad = clipped_grad + noise
                noisy_gradients.append(noisy_grad)
            else:
                noisy_gradients.append(grad)
        return noisy_gradients

# Quantization Class
class Quantization:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scale_factor = 2 ** (num_bits - 1) - 1
        
    def quantize_weights(self, weights):
        """Quantize model weights to reduce communication cost"""
        quantized_weights = []
        for w in weights:
            # Normalize to [-1, 1]
            w_max = tf.reduce_max(tf.abs(w))
            w_normalized = w / (w_max + 1e-8)
            
            # Quantize
            w_quantized = tf.round(w_normalized * self.scale_factor) / self.scale_factor
            
            # Restore scale
            w_restored = w_quantized * w_max
            quantized_weights.append(w_restored)
            
        return quantized_weights

# Fisher Information Calculator
class FisherInformationCalculator:
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        
    def calculate_fisher_information(self, model, x_data, y_data, batch_size=32):
        """Calculate Fisher Information Matrix for model parameters"""
        fisher_info = {}
        
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

# Partitioned Model for CBIS-DDSM
class PartitionedModel:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, client_id=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.client_id = client_id
        self.full_model = self._create_model()
        
    def _create_model(self):
        """Create a CNN model for Fashion-MNIST"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def get_weights(self):
        return self.full_model.get_weights()
    
    def set_weights(self, weights):
        self.full_model.set_weights(weights)
    
    def train_with_privacy(self, x_train, y_train, epochs=5, batch_size=32):
        """Train with differential privacy"""
        dp = DifferentialPrivacy(noise_multiplier=0.1)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        # Custom training loop with differential privacy
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(batch_size).shuffle(1000)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
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
                
                # Add differential privacy noise
                noisy_gradients = dp.add_noise_to_gradients(gradients)
                
                # Apply gradients
                optimizer.apply_gradients(zip(noisy_gradients, self.full_model.trainable_variables))
                
                epoch_loss += loss.numpy()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            print(f"Average epoch loss: {avg_loss:.4f}")
        
        return {'loss': epoch_loss}

# Initialize model and get partitioned data
model = PartitionedModel(client_id=client_id)

# Compile model
model.full_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"[CLIENT 1-{client_id}] Model compiled successfully")
print(f"[CLIENT 1-{client_id}] Total parameters: {model.full_model.count_params()}")

# Get partitioned data for this client
client_x_train, client_y_train, client_x_test, client_y_test = partition_data(
    "fashion_mnist", client_id, num_partitions=50
)

print(f"[CLIENT 1-{client_id}] Ready for federated learning with {len(client_x_train)} training samples")

# Define Flower client
class FashionMNISTFlowerClient50(fl.client.NumPyClient):
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
        
        # Train with differential privacy
        history = self.model.train_with_privacy(
            self.x_train, self.y_train, 
            epochs=config.get("local_epochs", 3),
            batch_size=32
        )
        
        print(f"[CLIENT 1-{client_id}] Training completed for round {current_round}")
        
        # Apply quantization to reduce communication cost
        quantizer = Quantization(num_bits=8)
        quantized_weights = quantizer.quantize_weights(self.model.get_weights())
        
        return quantized_weights, len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.full_model.evaluate(
            self.x_test, self.y_test, verbose=0
        )
        print(f"[CLIENT 1-{client_id}] Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.x_test), {"accuracy": accuracy}

# Create and start Flower client
client = FashionMNISTFlowerClient50(
    model, client_x_train, client_y_train, 
    client_x_test, client_y_test, client_id
)

print(f"[CLIENT 1-{client_id}] Starting Flower client...")
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)