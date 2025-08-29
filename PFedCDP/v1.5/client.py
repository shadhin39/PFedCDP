import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
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

client_id = int(input("Enter client ID for Model A (e.g., 1, 2, 3...): "))
print(f"\n=== INITIALIZING CLIENT A (ID: {client_id}) ===")

# Load dataset
x_train, y_train, x_test, y_test = partition_data(client_id)

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
        
        # KL divergence loss (teacher || student)
        kd_loss = tf.keras.losses.KLDivergence()(soft_teacher, soft_student)
        
        # Standard cross-entropy loss
        ce_loss = tf.keras.losses.categorical_crossentropy(
            true_labels_onehot, tf.nn.softmax(student_logits)
        )
        ce_loss = tf.reduce_mean(ce_loss)
        
        # Combined loss with temperature scaling
        total_loss = ((1 - self.lambda_balance) * ce_loss + 
                     self.lambda_balance * (self.temperature ** 2) * kd_loss)
        
        return total_loss, ce_loss, kd_loss

class PartitionedModel:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, client_id=None):
        self.client_id = client_id
        # Feature Extractor (W_L,k) - shared with server
        self.feature_extractor = keras.Sequential([
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu")
        ])
        
        # Client-specific Classifier (W_C,k) - NOT shared
        self.classifier = keras.Sequential([
            keras.Input(shape=(128,)),
            layers.Dense(num_classes, activation="softmax")
        ])
        
        # Full model for training
        self.full_model = keras.Sequential([
            self.feature_extractor,
            self.classifier
        ])
        
        # Teacher model (server's complete model)
        self.teacher_model = None
        
        # Knowledge distillation loss
        self.kd_loss = KnowledgeDistillationLoss(temperature=3.0, lambda_balance=0.3)
        
        # Epoch tracking
        self.epoch_metrics = []
    
    def get_feature_extractor_weights(self):
        return self.feature_extractor.get_weights()
    
    def set_feature_extractor_weights(self, weights):
        self.feature_extractor.set_weights(weights)
        
    def download_teacher_model(self, current_round):
        """Download teacher model weights from server"""
        teacher_model_path = f"Model_A_server_model-round-{current_round}.h5"
        
        if os.path.exists(teacher_model_path):
            try:
                # Load the complete server model
                self.teacher_model = keras.models.load_model(teacher_model_path, compile=False)
                print(f"[CLIENT A-{self.client_id}] Successfully loaded teacher model from {teacher_model_path}")
                return True
            except Exception as e:
                print(f"[CLIENT A-{self.client_id}] Error loading teacher model: {e}")
                return False
        else:
            print(f"[CLIENT A-{self.client_id}] Teacher model file {teacher_model_path} not found")
            return False
        
    def train_with_distillation(self, x_train, y_train, epochs=5, batch_size=64, current_round=1):
        """Train with knowledge distillation if teacher model is available"""
        if self.teacher_model is None:
            # Regular training without distillation - SHOW PROGRESS BAR
            print(f"\n[CLIENT A-{self.client_id}] Round {current_round}: Training without distillation")
            print(f"[CLIENT A-{self.client_id}] " + "="*60)
            
            # Custom callback for detailed epoch tracking
            class DetailedEpochTracker(keras.callbacks.Callback):
                def __init__(self, model_instance, round_num):
                    self.model_instance = model_instance
                    self.round_num = round_num
                    
                def on_epoch_begin(self, epoch, logs=None):
                    print(f"\n[CLIENT A-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs}")
                    
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    accuracy = logs.get('accuracy', 0.0)
                    loss = logs.get('loss', 0.0)
                    
                    # Store metrics
                    self.model_instance.epoch_metrics.append({
                        'round': self.round_num,
                        'epoch': epoch + 1,
                        'loss': float(loss),
                        'accuracy': float(accuracy),
                        'type': 'regular_training'
                    })
                    
                    # Log to file with detailed format
                    with open(f"client_A_epochs.txt", "a") as f:
                        f.write(f"[CLIENT A-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}\n")
                    
                    print(f"[CLIENT A-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")
            
            # Store epochs for callback access
            self.epochs = epochs
            epoch_tracker = DetailedEpochTracker(self, current_round)
            
            # Train with verbose=1 to show progress bars
            print(f"[CLIENT A-{self.client_id}] Starting training with progress bars...")
            self.full_model.fit(
                x_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                callbacks=[epoch_tracker],
                verbose=1  # Show progress bars
            )
            print(f"[CLIENT A-{self.client_id}] Training completed!")
            return
            
        # Custom training loop with knowledge distillation - SHOW DETAILED PROGRESS
        print(f"\n[CLIENT A-{self.client_id}] Round {current_round}: Training with knowledge distillation")
        print(f"[CLIENT A-{self.client_id}] " + "="*60)
        optimizer = keras.optimizers.Adam()
        
        # Convert labels to one-hot
        y_train_onehot = tf.one_hot(y_train, depth=10)
        
        # Create dataset with proper error handling
        try:
            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot))
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # Calculate total batches for progress display
            total_batches = len(x_train) // batch_size + (1 if len(x_train) % batch_size != 0 else 0)
            
            for epoch in range(epochs):
                print(f"\n[CLIENT A-{self.client_id}] Epoch {epoch + 1}/{epochs}")
                epoch_loss = 0
                epoch_ce_loss = 0
                epoch_kd_loss = 0
                epoch_accuracy = 0
                num_batches = 0
                
                # Progress tracking
                import time
                epoch_start_time = time.time()
                
                for batch_idx, (batch_x, batch_y) in enumerate(dataset):
                    batch_start_time = time.time()
                    
                    with tf.GradientTape() as tape:
                        # Student predictions (logits)
                        student_logits = self.full_model(batch_x, training=True)
                        
                        # Teacher predictions (logits)
                        teacher_logits = self.teacher_model(batch_x, training=False)
                        
                        # Compute distillation loss
                        total_loss, ce_loss, kd_loss = self.kd_loss.compute_distillation_loss(
                            student_logits, teacher_logits, tf.argmax(batch_y, axis=1)
                        )
                    
                    # Update student model
                    gradients = tape.gradient(total_loss, self.full_model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.full_model.trainable_variables))
                    
                    # Calculate accuracy
                    predictions = tf.argmax(student_logits, axis=1)
                    true_labels = tf.argmax(batch_y, axis=1)
                    batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, true_labels), tf.float32))
                    
                    epoch_loss += total_loss
                    epoch_ce_loss += ce_loss
                    epoch_kd_loss += kd_loss
                    epoch_accuracy += batch_accuracy
                    num_batches += 1
                    
                    # Show progress every 10 batches or at the end
                    batch_time = time.time() - batch_start_time
                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                        progress = "█" * int(20 * (batch_idx + 1) / total_batches)
                        remaining = "░" * (20 - int(20 * (batch_idx + 1) / total_batches))
                        print(f"\r[CLIENT A-{self.client_id}] {batch_idx + 1}/{total_batches} [{progress}{remaining}] - {batch_time*1000:.0f}ms/step", end="", flush=True)
                
                epoch_time = time.time() - epoch_start_time
                avg_loss = float(epoch_loss / num_batches)
                avg_ce_loss = float(epoch_ce_loss / num_batches)
                avg_kd_loss = float(epoch_kd_loss / num_batches)
                avg_accuracy = float(epoch_accuracy / num_batches)
                
                # Final epoch summary
                print(f"\n[CLIENT A-{self.client_id}] {epoch_time:.0f}s {epoch_time*1000/num_batches:.0f}ms/step - total_loss: {avg_loss:.4f} - ce_loss: {avg_ce_loss:.4f} - kd_loss: {avg_kd_loss:.4f} - accuracy: {avg_accuracy:.4f}")
                
                # Store metrics
                self.epoch_metrics.append({
                    'round': current_round,
                    'epoch': epoch + 1,
                    'total_loss': avg_loss,
                    'ce_loss': avg_ce_loss,
                    'kd_loss': avg_kd_loss,
                    'accuracy': avg_accuracy,
                    'type': 'distillation_training'
                })
                
                # Log to file with detailed format
                with open(f"client_A_epochs.txt", "a") as f:
                    f.write(f"[CLIENT A-{self.client_id}] Epoch {epoch + 1}/{epochs} - {epoch_time:.0f}s {epoch_time*1000/num_batches:.0f}ms/step - total_loss: {avg_loss:.4f} - ce_loss: {avg_ce_loss:.4f} - kd_loss: {avg_kd_loss:.4f} - accuracy: {avg_accuracy:.4f}\n")
                
        except tf.errors.OutOfRangeError:
            print(f"\n[CLIENT A-{self.client_id}] Dataset iteration completed successfully")
        except Exception as e:
            print(f"\n[CLIENT A-{self.client_id}] Training error: {e}")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.partitioned_model = PartitionedModel(client_id=client_id)
        self.partitioned_model.full_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        self.current_round = 1
        
        # Initialize epoch tracking file
        with open("client_A_epochs.txt", "w") as f:
            f.write(f"CLIENT A (ID: {client_id}) DETAILED EPOCH TRACKING\n")
            f.write("=" * 60 + "\n")
    
    def get_parameters(self, config):
        # Only return feature extractor weights
        return self.partitioned_model.get_feature_extractor_weights()
    
    def fit(self, parameters, config):
        print(f"\n{'='*70}")
        print(f"[CLIENT A-{self.client_id}] STARTING ROUND {self.current_round}")
        print(f"{'='*70}")
        
        # Set feature extractor weights from server
        self.partitioned_model.set_feature_extractor_weights(parameters)
        
        # Try to download teacher model from server
        teacher_available = self.partitioned_model.download_teacher_model(self.current_round)
        
        # Train with or without distillation
        self.partitioned_model.train_with_distillation(
            x_train, y_train, 
            epochs=5, 
            batch_size=64, 
            current_round=self.current_round
        )
        
        # Log round completion
        with open("client_A_epochs.txt", "a") as f:
            f.write(f"\n[CLIENT A-{self.client_id}] ROUND {self.current_round} COMPLETED\n" + "="*60 + "\n\n")
        
        print(f"\n{'='*70}")
        print(f"[CLIENT A-{self.client_id}] ROUND {self.current_round} COMPLETED")
        print(f"{'='*70}\n")
        
        # Increment round counter
        self.current_round += 1
        
        # Return only feature extractor weights
        return self.partitioned_model.get_feature_extractor_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        print(f"\n[CLIENT A-{self.client_id}] EVALUATING ROUND {self.current_round - 1}")
        
        # Set feature extractor weights from server
        self.partitioned_model.set_feature_extractor_weights(parameters)
        
        # Evaluate the full model with progress display
        try:
            print(f"[CLIENT A-{self.client_id}] Running evaluation...")
            loss, accuracy = self.partitioned_model.full_model.evaluate(x_test, y_test, verbose=1)
            
            # Log evaluation results
            with open("client_A_epochs.txt", "a") as f:
                f.write(f"[CLIENT A-{self.client_id}] Evaluation Round {self.current_round - 1}: Loss={loss:.4f}, Accuracy={accuracy:.4f}\n")
            
            print(f"[CLIENT A-{self.client_id}] Evaluation Results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            return float(loss), len(x_test), {"accuracy": float(accuracy)}
            
        except Exception as e:
            print(f"[CLIENT A-{self.client_id}] Evaluation error: {e}")
            return 0.0, len(x_test), {"accuracy": 0.0}

# Connect to server on port 8080
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(client_id))