import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
# Import the extracted components
from data_loader import partition_data
from differential_privacy import DifferentialPrivacy
from quantization import Quantization
from fisher_information_calculation import FisherInformationCalculator
from knowledge_distillation_client import KnowledgeDistillationLoss
# Add new imports at the top
from dataset_loaders import load_cbis_ddsm_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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

class DifferentialPrivacy:
    """Differential Privacy implementation for federated learning"""
    
    def __init__(self, epsilon=8.0, delta=1e-5, clipping_threshold=2.5, noise_multiplier=None):
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_threshold = clipping_threshold
        self.noise_multiplier = noise_multiplier or self._compute_noise_multiplier()
    
    def _compute_noise_multiplier(self):
        """Compute noise multiplier based on privacy parameters"""
        # Simplified noise multiplier calculation
        # In practice, use privacy accountant for precise calculation
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_weights(self, weights):
        """Clip weights using L2 norm clipping"""
        clipped_weights = []
        for w in weights:
            # Calculate L2 norm
            l2_norm = tf.norm(w, ord=2)
            # Apply clipping
            clipping_factor = tf.minimum(1.0, self.clipping_threshold / l2_norm)
            clipped_w = w * clipping_factor
            clipped_weights.append(clipped_w)
        return clipped_weights
    
    def add_noise(self, weights):
        """Add calibrated Gaussian noise for differential privacy"""
        noisy_weights = []
        for w in weights:
            # Generate Gaussian noise
            noise_stddev = self.clipping_threshold * self.noise_multiplier
            noise = tf.random.normal(shape=w.shape, mean=0.0, stddev=noise_stddev)
            noisy_w = w + noise
            noisy_weights.append(noisy_w)
        return noisy_weights
    
    def apply_dp(self, weights):
        """Apply differential privacy: clipping + noise addition"""
        print(f"[DP] Applying differential privacy with ε={self.epsilon}, δ={self.delta}")
        
        # Step 1: Clip weights
        clipped_weights = self.clip_weights(weights)
        
        # Step 2: Add noise
        dp_weights = self.add_noise(clipped_weights)
        
        print(f"[DP] Differential privacy applied successfully")
        return dp_weights

class Quantization:
    """Quantization implementation for communication efficiency"""
    
    def __init__(self, num_bits=12):
        self.num_bits = num_bits
        self.max_val = 2**num_bits - 1
    
    def quantize_weights(self, weights):
        """Quantize weights to lower-bit representation"""
        print(f"[QUANTIZATION] Quantizing weights to {self.num_bits} bits")
        
        quantized_weights = []
        quantization_params = []
        
        for w in weights:
            # Find min and max values
            w_min = tf.reduce_min(w)
            w_max = tf.reduce_max(w)
            
            # Quantize: scale to [0, 2^b - 1] and round
            w_scaled = (w - w_min) / (w_max - w_min + 1e-8)  # Scale to [0, 1]
            w_quantized = tf.round(w_scaled * self.max_val)  # Scale to [0, 2^b-1] and round
            
            # Convert to integers
            w_quantized = tf.cast(w_quantized, tf.int32)
            
            quantized_weights.append(w_quantized.numpy())
            quantization_params.append({
                'min': float(w_min.numpy()),
                'max': float(w_max.numpy())
            })
        
        print(f"[QUANTIZATION] Quantization completed")
        return quantized_weights, quantization_params

client_id = int(input("Enter client ID for Model B (e.g., 1, 2, 3...): "))
print(f"\n=== INITIALIZING CLIENT B (ID: {client_id}) ===")

# Load CBIS-DDSM dataset instead of Fashion-MNIST
x_train, y_train, x_test, y_test = load_cbis_ddsm_dataset()

# Knowledge Distillation Loss Function
class KnowledgeDistillationLoss:
    def __init__(self, temperature=3.0, lambda_balance=0.5):
        self.temperature = temperature
        self.lambda_balance = lambda_balance
        
    def compute_distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Compute combined distillation and CE loss"""
        # Convert true labels to one-hot if needed
        if len(true_labels.shape) == 1:
            true_labels_onehot = tf.one_hot(true_labels, depth=2)  # Changed from 10 to 2
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

class FisherInformationCalculator:
    """Calculate Fisher Information Matrix for feature extractor personalization"""
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
    
    class FisherInformationCalculator:
        def calculate_fisher_information(self, model, x_data, y_data, batch_size=32):
            """Calculate Fisher Information Matrix for model parameters"""
            print(f"[FISHER] Calculating Fisher Information Matrix...")
            
            # Ensure proper data format
            if len(y_data.shape) == 1:
                y_data_onehot = tf.one_hot(y_data, depth=2)  # Changed from 10 to 2
            else:
                y_data_onehot = y_data
            
            # Create dataset with proper error handling
            dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data_onehot))
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # Initialize Fisher information storage
            fisher_info = []
            for layer in model.trainable_variables:
                fisher_info.append(tf.zeros_like(layer))
            
            num_samples = 0
            
            # Calculate Fisher information over batches with proper error handling
            try:
                for batch_x, batch_y in dataset:
                    try:
                        with tf.GradientTape() as tape:
                            predictions = model(batch_x, training=False)
                            loss = tf.keras.losses.categorical_crossentropy(batch_y, predictions)
                            loss = tf.reduce_mean(loss)
                        
                        # Calculate gradients
                        gradients = tape.gradient(loss, model.trainable_variables)
                        
                        # Accumulate squared gradients (Fisher information)
                        for i, grad in enumerate(gradients):
                            if grad is not None:
                                fisher_info[i] += tf.square(grad) * tf.cast(batch_x.shape[0], tf.float32)
                        
                        num_samples += batch_x.shape[0]
                        
                    except tf.errors.OutOfRangeError:
                        print(f"[FISHER] Reached end of batch, continuing...")
                        break
                    except Exception as e:
                        print(f"[FISHER] Batch processing error: {e}, skipping batch")
                        continue
                        
            except tf.errors.OutOfRangeError:
                print(f"[FISHER] Dataset iteration completed")
            except Exception as e:
                print(f"[FISHER] Dataset error: {e}")
            
            # Normalize by number of samples
            if num_samples > 0:
                for i in range(len(fisher_info)):
                    fisher_info[i] /= float(num_samples)
            else:
                print(f"[FISHER] Warning: No samples processed, using zero Fisher information")
            
            print(f"[FISHER] Fisher Information calculated for {num_samples} samples")
            return fisher_info
        
        def create_binary_masks(self, fisher_info):
            """Create personal and global masks based on Fisher information"""
            personal_masks = []
            global_masks = []
            
            total_params = 0
            high_fisher_params = 0
            
            for fisher_layer in fisher_info:
                # Calculate mean Fisher value for the layer
                mean_fisher = tf.reduce_mean(fisher_layer)
                
                # Create masks based on threshold
                personal_mask = tf.cast(fisher_layer >= self.threshold, tf.float32)
                global_mask = tf.cast(fisher_layer < self.threshold, tf.float32)
                
                personal_masks.append(personal_mask)
                global_masks.append(global_mask)
                
                # Statistics
                layer_total = tf.size(fisher_layer).numpy()
                layer_high_fisher = tf.reduce_sum(personal_mask).numpy()
                
                total_params += layer_total
                high_fisher_params += layer_high_fisher
                
                print(f"[FISHER] Layer shape {fisher_layer.shape}: {layer_high_fisher}/{layer_total} ({100*layer_high_fisher/layer_total:.1f}%) high Fisher params")
            
            print(f"[FISHER] Total: {high_fisher_params}/{total_params} ({100*high_fisher_params/total_params:.1f}%) parameters kept locally")
            
            return personal_masks, global_masks
        
        def apply_fisher_personalization(self, local_weights, global_weights, personal_masks, global_masks):
            """Apply Fisher-based personalization to combine local and global weights"""
            personalized_weights = []
            
            for i, (local_w, global_w, personal_mask, global_mask) in enumerate(
                zip(local_weights, global_weights, personal_masks, global_masks)):
                
            # Combine weights using masks: W_L,k^t = M_personal ⊙ W_L,k^{t-1} + M_global ⊙ W_L,global^{t-1}
            personalized_w = personal_mask * local_w + global_mask * global_w
            personalized_weights.append(personalized_w)
        
            return personalized_weights
    
    # Partitioned Model with lighter architecture (matching client.py structure but lighter)
    class PartitionedModel:
        def __init__(self, input_shape=(224, 224, 1), num_classes=2, client_id=None):  # Changed input shape and classes
            self.client_id = client_id
            # Feature Extractor (W_L,k) - shared with server (adapted for mammography)
            self.feature_extractor = keras.Sequential([
                keras.Input(shape=input_shape),  # (224, 224, 1) for CBIS-DDSM
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.GlobalAveragePooling2D(),  # Better for larger images
                layers.Dense(512, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(256, activation="relu")
            ])
            
            # Client-specific Classifier (W_C,k) - NOT shared (2 classes for CBIS-DDSM)
            self.classifier = keras.Sequential([
                keras.Input(shape=(256,)),  # Adjusted input shape
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")  # 2 classes: benign/malignant
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
            
            # Fisher Information Calculator
            self.fisher_calculator = FisherInformationCalculator(threshold=0.01)
            
            # Store Fisher masks for reuse
            self.personal_masks = None
            self.global_masks = None
            
            # Epoch tracking
            self.epoch_metrics = []
        
        def get_feature_extractor_weights(self):
            return self.feature_extractor.get_weights()
        
        def set_feature_extractor_weights(self, weights):
            self.feature_extractor.set_weights(weights)
        
        def apply_fisher_personalization(self, global_weights, x_train, y_train):
            """Apply Fisher Information-based personalization to feature extractor"""
            print(f"\n[CLIENT B-{self.client_id}] Applying Fisher Information personalization...")
            
            # Get current local weights
            local_weights = self.get_feature_extractor_weights()
            
            # Calculate Fisher information if not already done
            if self.personal_masks is None or self.global_masks is None:
                print(f"[CLIENT B-{self.client_id}] Computing Fisher Information Matrix...")
                
                # Use a subset of training data for Fisher calculation (for efficiency)
                subset_size = min(1000, len(x_train))
                indices = np.random.choice(len(x_train), subset_size, replace=False)
                x_subset = x_train[indices]
                y_subset = y_train[indices]
                
                # Calculate Fisher information using the FULL MODEL, not just feature extractor
                fisher_info = self.fisher_calculator.calculate_fisher_information(
                    self.full_model, x_subset, y_subset  # Changed from self.feature_extractor to self.full_model
                )
                
                # But only keep Fisher info for feature extractor layers
                # The full model has feature extractor + classifier layers
                # We only want Fisher info for feature extractor layers
                num_feature_layers = len(self.feature_extractor.trainable_variables)
                fisher_info = fisher_info[:num_feature_layers]  # Keep only feature extractor Fisher info
                
                # Create binary masks
                self.personal_masks, self.global_masks = self.fisher_calculator.create_binary_masks(fisher_info)
            
            # Apply personalization
            personalized_weights = self.fisher_calculator.apply_fisher_personalization(
                local_weights, global_weights, self.personal_masks, self.global_masks
            )
            
            # Set the personalized weights
            self.set_feature_extractor_weights(personalized_weights)
            
            print(f"[CLIENT B-{self.client_id}] Fisher personalization applied successfully")
            
        def download_teacher_model(self, current_round):
            """Download teacher model weights from server"""
            teacher_model_path = f"Model_B_server_model-round-{current_round}.h5"
            
            if os.path.exists(teacher_model_path):
                try:
                    # Load the complete server model
                    self.teacher_model = keras.models.load_model(teacher_model_path, compile=False)
                    print(f"[CLIENT B-{self.client_id}] Successfully loaded teacher model from {teacher_model_path}")
                    return True
                except Exception as e:
                    print(f"[CLIENT B-{self.client_id}] Error loading teacher model: {e}")
                    return False
            else:
                print(f"[CLIENT B-{self.client_id}] Teacher model file {teacher_model_path} not found")
                return False
            
        def train_with_distillation(self, x_train, y_train, epochs=5, batch_size=64, current_round=1):
            """Train with knowledge distillation if teacher model is available"""
            if self.teacher_model is None:
                # Regular training without distillation - SHOW PROGRESS BAR
                print(f"\n[CLIENT B-{self.client_id}] Round {current_round}: Training without distillation")
                print(f"[CLIENT B-{self.client_id}] " + "="*60)
                
                # Custom callback for detailed epoch tracking
                class DetailedEpochTracker(keras.callbacks.Callback):
                    def __init__(self, model_instance, round_num):
                        self.model_instance = model_instance
                        self.round_num = round_num
                        
                    def on_epoch_begin(self, epoch, logs=None):
                        print(f"\n[CLIENT B-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs}")
                        
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
                        with open(f"client_B_epochs.txt", "a") as f:
                            f.write(f"[CLIENT B-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}\n")
                        
                        print(f"[CLIENT B-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")
                
                # Store epochs for callback access
                self.epochs = epochs
                epoch_tracker = DetailedEpochTracker(self, current_round)
                
                # Train with verbose=1 to show progress bars
                print(f"[CLIENT B-{self.client_id}] Starting training with progress bars...")
                self.full_model.fit(
                    x_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    callbacks=[epoch_tracker],
                    verbose=1  # Show progress bars
                )
                print(f"[CLIENT B-{self.client_id}] Training completed!")
                return
                
            # Custom training loop with knowledge distillation - SHOW DETAILED PROGRESS
            print(f"\n[CLIENT B-{self.client_id}] Round {current_round}: Training with knowledge distillation")
            print(f"[CLIENT B-{self.client_id}] " + "="*60)
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
                    print(f"\n[CLIENT B-{self.client_id}] Epoch {epoch + 1}/{epochs}")
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
                            print(f"\r[CLIENT B-{self.client_id}] {batch_idx + 1}/{total_batches} [{progress}{remaining}] - {batch_time*1000:.0f}ms/step", end="", flush=True)
                    
                    epoch_time = time.time() - epoch_start_time
                    avg_loss = float(epoch_loss / num_batches)
                    avg_ce_loss = float(epoch_ce_loss / num_batches)
                    avg_kd_loss = float(epoch_kd_loss / num_batches)
                    avg_accuracy = float(epoch_accuracy / num_batches)
                    
                    # Final epoch summary
                    print(f"\n[CLIENT B-{self.client_id}] {epoch_time:.0f}s {epoch_time*1000/num_batches:.0f}ms/step - total_loss: {avg_loss:.4f} - ce_loss: {avg_ce_loss:.4f} - kd_loss: {avg_kd_loss:.4f} - accuracy: {avg_accuracy:.4f}")
                    
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
                    with open(f"client_B_epochs.txt", "a") as f:
                        f.write(f"[CLIENT B-{self.client_id}] Epoch {epoch + 1}/{epochs} - {epoch_time:.0f}s {epoch_time*1000/num_batches:.0f}ms/step - total_loss: {avg_loss:.4f} - ce_loss: {avg_ce_loss:.4f} - kd_loss: {avg_kd_loss:.4f} - accuracy: {avg_accuracy:.4f}\n")
                        
            except tf.errors.OutOfRangeError:
                print(f"\n[CLIENT B-{self.client_id}] Dataset iteration completed successfully")
            except Exception as e:
                print(f"\n[CLIENT B-{self.client_id}] Training error: {e}")

client_id = int(input("Enter client ID for Model B (e.g., 1, 2, 3...): "))
print(f"\n=== INITIALIZING CLIENT B (ID: {client_id}) ===")

# Load CBIS-DDSM dataset instead of Fashion-MNIST
x_train, y_train, x_test, y_test = load_cbis_ddsm_dataset()

# Knowledge Distillation Loss Function
class KnowledgeDistillationLoss:
    def __init__(self, temperature=3.0, lambda_balance=0.5):
        self.temperature = temperature
        self.lambda_balance = lambda_balance
        
    def compute_distillation_loss(self, student_logits, teacher_logits, true_labels):
        """Compute combined distillation and CE loss"""
        # Convert true labels to one-hot if needed
        if len(true_labels.shape) == 1:
            true_labels_onehot = tf.one_hot(true_labels, depth=2)  # Changed from 10 to 2
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

class FisherInformationCalculator:
    """Calculate Fisher Information Matrix for feature extractor personalization"""
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
    
    def calculate_fisher_information(self, model, x_data, y_data, batch_size=32):
        """Calculate Fisher Information Matrix for model parameters"""
        print(f"[FISHER] Calculating Fisher Information Matrix...")
        
        # Ensure proper data format
        if len(y_data.shape) == 1:
            y_data_onehot = tf.one_hot(y_data, depth=2)  # Changed from 10 to 2
        else:
            y_data_onehot = y_data
        
        # Create dataset with proper error handling
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data_onehot))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Initialize Fisher information storage
        fisher_info = []
        for layer in model.trainable_variables:
            fisher_info.append(tf.zeros_like(layer))
        
        num_samples = 0
        
        # Calculate Fisher information over batches with proper error handling
        try:
            for batch_x, batch_y in dataset:
                try:
                    with tf.GradientTape() as tape:
                        predictions = model(batch_x, training=False)
                        loss = tf.keras.losses.categorical_crossentropy(batch_y, predictions)
                        loss = tf.reduce_mean(loss)
                    
                    # Calculate gradients
                    gradients = tape.gradient(loss, model.trainable_variables)
                    
                    # Accumulate squared gradients (Fisher information)
                    for i, grad in enumerate(gradients):
                        if grad is not None:
                            fisher_info[i] += tf.square(grad) * tf.cast(batch_x.shape[0], tf.float32)
                    
                    num_samples += batch_x.shape[0]
                    
                except tf.errors.OutOfRangeError:
                    print(f"[FISHER] Reached end of batch, continuing...")
                    break
                except Exception as e:
                    print(f"[FISHER] Batch processing error: {e}, skipping batch")
                    continue
                    
        except tf.errors.OutOfRangeError:
            print(f"[FISHER] Dataset iteration completed")
        except Exception as e:
            print(f"[FISHER] Dataset error: {e}")
        
        # Normalize by number of samples
        if num_samples > 0:
            for i in range(len(fisher_info)):
                fisher_info[i] /= float(num_samples)
        else:
            print(f"[FISHER] Warning: No samples processed, using zero Fisher information")
        
        print(f"[FISHER] Fisher Information calculated for {num_samples} samples")
        return fisher_info
    
    def create_binary_masks(self, fisher_info):
        """Create personal and global masks based on Fisher information"""
        personal_masks = []
        global_masks = []
        
        total_params = 0
        high_fisher_params = 0
        
        for fisher_layer in fisher_info:
            # Calculate mean Fisher value for the layer
            mean_fisher = tf.reduce_mean(fisher_layer)
            
            # Create masks based on threshold
            personal_mask = tf.cast(fisher_layer >= self.threshold, tf.float32)
            global_mask = tf.cast(fisher_layer < self.threshold, tf.float32)
            
            personal_masks.append(personal_mask)
            global_masks.append(global_mask)
            
            # Statistics
            layer_total = tf.size(fisher_layer).numpy()
            layer_high_fisher = tf.reduce_sum(personal_mask).numpy()
            
            total_params += layer_total
            high_fisher_params += layer_high_fisher
            
            print(f"[FISHER] Layer shape {fisher_layer.shape}: {layer_high_fisher}/{layer_total} ({100*layer_high_fisher/layer_total:.1f}%) high Fisher params")
        
        print(f"[FISHER] Total: {high_fisher_params}/{total_params} ({100*high_fisher_params/total_params:.1f}%) parameters kept locally")
        
        return personal_masks, global_masks
    
    def apply_fisher_personalization(self, local_weights, global_weights, personal_masks, global_masks):
        """Apply Fisher-based personalization to combine local and global weights"""
        personalized_weights = []
        
        for i, (local_w, global_w, personal_mask, global_mask) in enumerate(
            zip(local_weights, global_weights, personal_masks, global_masks)):
            
            # Combine weights using masks: W_L,k^t = M_personal ⊙ W_L,k^{t-1} + M_global ⊙ W_L,global^{t-1}
            personalized_w = personal_mask * local_w + global_mask * global_w
            personalized_weights.append(personalized_w)
        
        return personalized_weights

# Partitioned Model with lighter architecture (matching client.py structure but lighter)
class PartitionedModel:
    def __init__(self, input_shape=(224, 224, 1), num_classes=2, client_id=None):  # Changed input shape and classes
        self.client_id = client_id
        # Feature Extractor (W_L,k) - shared with server (adapted for mammography)
        self.feature_extractor = keras.Sequential([
            keras.Input(shape=input_shape),  # (224, 224, 1) for CBIS-DDSM
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.GlobalAveragePooling2D(),  # Better for larger images
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu")
        ])
        
        # Client-specific Classifier (W_C,k) - NOT shared (2 classes for CBIS-DDSM)
        self.classifier = keras.Sequential([
            keras.Input(shape=(256,)),  # Adjusted input shape
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")  # 2 classes: benign/malignant
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
        
        # Fisher Information Calculator
        self.fisher_calculator = FisherInformationCalculator(threshold=0.01)
        
        # Store Fisher masks for reuse
        self.personal_masks = None
        self.global_masks = None
        
        # Epoch tracking
        self.epoch_metrics = []
    
    def get_feature_extractor_weights(self):
        return self.feature_extractor.get_weights()
    
    def set_feature_extractor_weights(self, weights):
        self.feature_extractor.set_weights(weights)
    
    def apply_fisher_personalization(self, global_weights, x_train, y_train):
        """Apply Fisher Information-based personalization to feature extractor"""
        print(f"\n[CLIENT B-{self.client_id}] Applying Fisher Information personalization...")
        
        # Get current local weights
        local_weights = self.get_feature_extractor_weights()
        
        # Calculate Fisher information if not already done
        if self.personal_masks is None or self.global_masks is None:
            print(f"[CLIENT B-{self.client_id}] Computing Fisher Information Matrix...")
            
            # Use a subset of training data for Fisher calculation (for efficiency)
            subset_size = min(1000, len(x_train))
            indices = np.random.choice(len(x_train), subset_size, replace=False)
            x_subset = x_train[indices]
            y_subset = y_train[indices]
            
            # Calculate Fisher information using the FULL MODEL, not just feature extractor
            fisher_info = self.fisher_calculator.calculate_fisher_information(
                self.full_model, x_subset, y_subset  # Changed from self.feature_extractor to self.full_model
            )
            
            # But only keep Fisher info for feature extractor layers
            # The full model has feature extractor + classifier layers
            # We only want Fisher info for feature extractor layers
            num_feature_layers = len(self.feature_extractor.trainable_variables)
            fisher_info = fisher_info[:num_feature_layers]  # Keep only feature extractor Fisher info
            
            # Create binary masks
            self.personal_masks, self.global_masks = self.fisher_calculator.create_binary_masks(fisher_info)
        
        # Apply personalization
        personalized_weights = self.fisher_calculator.apply_fisher_personalization(
            local_weights, global_weights, self.personal_masks, self.global_masks
        )
        
        # Set the personalized weights
        self.set_feature_extractor_weights(personalized_weights)
        
        print(f"[CLIENT B-{self.client_id}] Fisher personalization applied successfully")
        
    def download_teacher_model(self, current_round):
        """Download teacher model weights from server"""
        teacher_model_path = f"Model_B_server_model-round-{current_round}.h5"
        
        if os.path.exists(teacher_model_path):
            try:
                # Load the complete server model
                self.teacher_model = keras.models.load_model(teacher_model_path, compile=False)
                print(f"[CLIENT B-{self.client_id}] Successfully loaded teacher model from {teacher_model_path}")
                return True
            except Exception as e:
                print(f"[CLIENT B-{self.client_id}] Error loading teacher model: {e}")
                return False
        else:
            print(f"[CLIENT B-{self.client_id}] Teacher model file {teacher_model_path} not found")
            return False
        
    def train_with_distillation(self, x_train, y_train, epochs=5, batch_size=64, current_round=1):
        """Train with knowledge distillation if teacher model is available"""
        if self.teacher_model is None:
            # Regular training without distillation - SHOW PROGRESS BAR
            print(f"\n[CLIENT B-{self.client_id}] Round {current_round}: Training without distillation")
            print(f"[CLIENT B-{self.client_id}] " + "="*60)
            
            # Custom callback for detailed epoch tracking
            class DetailedEpochTracker(keras.callbacks.Callback):
                def __init__(self, model_instance, round_num):
                    self.model_instance = model_instance
                    self.round_num = round_num
                    
                def on_epoch_begin(self, epoch, logs=None):
                    print(f"\n[CLIENT B-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs}")
                    
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
                    with open(f"client_B_epochs.txt", "a") as f:
                        f.write(f"[CLIENT B-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}\n")
                    
                    print(f"[CLIENT B-{self.model_instance.client_id}] Epoch {epoch + 1}/{self.model_instance.epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")
            
            # Store epochs for callback access
            self.epochs = epochs
            epoch_tracker = DetailedEpochTracker(self, current_round)
            
            # Train with verbose=1 to show progress bars
            print(f"[CLIENT B-{self.client_id}] Starting training with progress bars...")
            self.full_model.fit(
                x_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                callbacks=[epoch_tracker],
                verbose=1  # Show progress bars
            )
            print(f"[CLIENT B-{self.client_id}] Training completed!")
            return
            
        # Custom training loop with knowledge distillation - SHOW DETAILED PROGRESS
        print(f"\n[CLIENT B-{self.client_id}] Round {current_round}: Training with knowledge distillation")
        print(f"[CLIENT B-{self.client_id}] " + "="*60)
        optimizer = keras.optimizers.Adam()
        
        # Convert labels to one-hot
        y_train_onehot = tf.one_hot(y_train, depth=2)
        
        # Create dataset with proper error handling
        try:
            dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot))
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            # Calculate total batches for progress display
            total_batches = len(x_train) // batch_size + (1 if len(x_train) % batch_size != 0 else 0)
            
            for epoch in range(epochs):
                print(f"\n[CLIENT B-{self.client_id}] Epoch {epoch + 1}/{epochs}")
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
                        print(f"\r[CLIENT B-{self.client_id}] {batch_idx + 1}/{total_batches} [{progress}{remaining}] - {batch_time*1000:.0f}ms/step", end="", flush=True)
                
                epoch_time = time.time() - epoch_start_time
                avg_loss = float(epoch_loss / num_batches)
                avg_ce_loss = float(epoch_ce_loss / num_batches)
                avg_kd_loss = float(epoch_kd_loss / num_batches)
                avg_accuracy = float(epoch_accuracy / num_batches)
                
                # Final epoch summary
                print(f"\n[CLIENT B-{self.client_id}] {epoch_time:.0f}s {epoch_time*1000/num_batches:.0f}ms/step - total_loss: {avg_loss:.4f} - ce_loss: {avg_ce_loss:.4f} - kd_loss: {avg_kd_loss:.4f} - accuracy: {avg_accuracy:.4f}")
                
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
                with open(f"client_B_epochs.txt", "a") as f:
                    f.write(f"[CLIENT B-{self.client_id}] Epoch {epoch + 1}/{epochs} - {epoch_time:.0f}s {epoch_time*1000/num_batches:.0f}ms/step - total_loss: {avg_loss:.4f} - ce_loss: {avg_ce_loss:.4f} - kd_loss: {avg_kd_loss:.4f} - accuracy: {avg_accuracy:.4f}\n")
                
        except tf.errors.OutOfRangeError:
            print(f"\n[CLIENT B-{self.client_id}] Dataset iteration completed successfully")
        except Exception as e:
            print(f"\n[CLIENT B-{self.client_id}] Training error: {e}")

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
        with open(f"client_B_{client_id}_epochs.txt", "w") as f:
            f.write(f"CLIENT B (ID: {client_id}) DETAILED EPOCH TRACKING\n")
            f.write("=" * 60 + "\n")

    def get_parameters(self, config):
        return self.partitioned_model.get_feature_extractor_weights()

    def fit(self, parameters, config):
        print(f"\n[CLIENT B-{self.client_id}] TRAINING ROUND {self.current_round}")
        
        # Set feature extractor weights from server
        self.partitioned_model.set_feature_extractor_weights(parameters)
        
        # Apply differential privacy and quantization
        dp = DifferentialPrivacy(epsilon=8.0, delta=1e-5)
        quantization = Quantization(num_bits=12)
        
        # Apply Fisher Information personalization
        self.partitioned_model.apply_fisher_personalization(parameters, x_train, y_train)
        
        # Train with knowledge distillation
        self.partitioned_model.train_with_distillation(
            x_train, y_train, epochs=5, batch_size=64, current_round=self.current_round
        )
        
        # Apply differential privacy to weights
        weights = self.partitioned_model.get_feature_extractor_weights()
        weights = dp.apply_dp(weights)
        weights = quantization.quantize_weights(weights)
        
        print(f"\n{'='*70}")
        print(f"[CLIENT B-{self.client_id}] ROUND {self.current_round} COMPLETED")
        print(f"{'='*70}\n")
        
        # Increment round counter
        self.current_round += 1
        
        return weights, len(x_train), {}

    def evaluate(self, parameters, config):
        print(f"\n[CLIENT B-{self.client_id}] EVALUATING ROUND {self.current_round - 1}")
        
        # Set feature extractor weights from server
        self.partitioned_model.set_feature_extractor_weights(parameters)
        
        # Evaluate the full model
        try:
            print(f"[CLIENT B-{self.client_id}] Running evaluation...")
            loss, accuracy = self.partitioned_model.full_model.evaluate(x_test, y_test, verbose=1)
            
            # Get predictions for additional metrics
            y_pred_probs = self.partitioned_model.full_model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Calculate comprehensive metrics
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Calculate AUC (handle binary vs multi-class)
            try:
                if len(np.unique(y_test)) == 2:
                    auc = roc_auc_score(y_test, y_pred_probs[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr', average='weighted')
            except ValueError:
                auc = 0.0
            
            # Calculate specificity and sensitivity from confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape[0] == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            else:  # Multi-class classification
                specificity_list = []
                sensitivity_list = []
                for i in range(cm.shape[0]):
                    tp = cm[i, i]
                    fn = np.sum(cm[i, :]) - tp
                    fp = np.sum(cm[:, i]) - tp
                    tn = np.sum(cm) - tp - fn - fp
                    
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    specificity_list.append(spec)
                    sensitivity_list.append(sens)
                
                specificity = np.mean(specificity_list)
                sensitivity = np.mean(sensitivity_list)
            
            # Log evaluation results
            with open(f"client_B_{self.client_id}_epochs.txt", "a") as f:
                f.write(f"[CLIENT B-{self.client_id}] Evaluation Round {self.current_round - 1}:\n")
                f.write(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
                f.write(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
                f.write(f"  AUC: {auc:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}\n")
            
            print(f"[CLIENT B-{self.client_id}] Evaluation Results:")
            print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"  AUC: {auc:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}")
            
            return float(loss), len(x_test), {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc": float(auc),
                "specificity": float(specificity),
                "sensitivity": float(sensitivity)
            }
            
        except Exception as e:
            print(f"[CLIENT B-{self.client_id}] Evaluation error: {e}")
            return 0.0, len(x_test), {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0
            }

# Connect to server
fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=FlowerClient(client_id=1))
