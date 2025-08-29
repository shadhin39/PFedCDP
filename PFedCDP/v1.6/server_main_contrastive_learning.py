import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import multiprocessing
import os
import h5py
from calculate_feature_importance import calculate_feature_importance
# Add Excel export libraries
import pandas as pd
from datetime import datetime
import openpyxl

# Global tracking variables
accuraciess_A = []
losses_A = []
aggregated_weight_A = None
aggregated_weight_B = None
aggregated_weight_C = None  # Added for Model C
federated_rounds = 10
accuraciess_B = []
losses_B = []
accuraciess_C = []  # Added for Model C
losses_C = []      # Added for Model C

# Enhanced epoch tracking with detailed metrics
epoch_metrics = {
    "Model_A": {
        "rounds": [], "accuracies": [], "losses": [], 
        "client_epochs": [], "server_epochs": []
    },
    "Model_B": {
        "rounds": [], "accuracies": [], "losses": [], 
        "client_epochs": [], "server_epochs": []
    },
    "Model_C": {  # Added for Model C
        "rounds": [], "accuracies": [], "losses": [], 
        "client_epochs": [], "server_epochs": []
    }
}

##############################################################################
# Enhanced Multi-Teacher Distillation Model
##############################################################################
class MultiTeacherDistiller(keras.Model):
    def __init__(self, student, teachers, feature_importance, embedding_dim=256, epsilon=0.01, current_epoch=1):
        super().__init__()
        self.teachers = teachers
        self.student = student
        self.feature_importance = feature_importance
        self.epsilon = epsilon
        self.current_epoch = current_epoch
        self.temperature = 0.1
        
        # Build the student model first
        dummy_input = tf.zeros((1, 28, 28, 1))
        _ = self.student(dummy_input)
        
        # Enhanced projection head
        self.projection = keras.Sequential([
            layers.Dense(embedding_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(embedding_dim // 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(embedding_dim // 4)
        ])
        
        # Feature extractor from student model
        feature_layers = self.student.layers[:-1]
        self.feature_extractor = keras.Sequential(feature_layers)

    def compile(self, optimizer, metrics=None, distillation_loss_fn=None, alpha=0.3, temperature=0.1):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def calculate_improved_shapley_values(self, embeddings, labels):
        """Calculate improved Shapley values with proper error handling"""
        try:
            embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
            batch_size = tf.shape(embeddings)[0]
            labels = tf.cast(labels, tf.int32)
            shapley_values = tf.zeros(batch_size, dtype=tf.float32)
            
            class_indices = tf.range(10, dtype=tf.int32)
            
            def fold_fn(accumulated_shapley, class_idx):
                class_label = tf.cast(class_idx, tf.int32)
                mask = tf.equal(labels, class_label)
                
                def compute_class_shapley():
                    class_embeddings = tf.boolean_mask(embeddings_norm, mask)
                    centroid = tf.reduce_mean(class_embeddings, axis=0)
                    own_distances = tf.reduce_sum(tf.square(class_embeddings - centroid), axis=1)
                    
                    other_mask = tf.logical_not(mask)
                    
                    def compute_other_distances():
                        other_embeddings = tf.boolean_mask(embeddings_norm, other_mask)
                        other_centroid = tf.reduce_mean(other_embeddings, axis=0)
                        other_distances = tf.reduce_sum(tf.square(class_embeddings - other_centroid), axis=1)
                        return other_distances / (own_distances + 1e-8)
                    
                    def default_shapley():
                        return tf.ones_like(own_distances)
                    
                    class_shapley = tf.cond(
                        tf.reduce_any(other_mask),
                        compute_other_distances,
                        default_shapley
                    )
                    
                    indices = tf.where(mask)
                    return tf.tensor_scatter_nd_update(accumulated_shapley, indices, class_shapley)
                
                def no_update():
                    return accumulated_shapley
                
                return tf.cond(
                    tf.reduce_any(mask),
                    compute_class_shapley,
                    no_update
                )
            
            shapley_values = tf.foldl(
                fold_fn,
                class_indices,
                initializer=shapley_values,
                parallel_iterations=1
            )
            
            shapley_values = tf.nn.softmax(shapley_values)
            return shapley_values
        except Exception as e:
            tf.print(f"Error in Shapley calculation: {e}")
            return tf.ones(tf.shape(embeddings)[0], dtype=tf.float32) / tf.cast(tf.shape(embeddings)[0], tf.float32)

    def improved_supervised_contrastive_loss(self, embeddings, labels):
        """Enhanced contrastive loss with better error handling"""
        try:
            embeddings = tf.nn.l2_normalize(embeddings, axis=1)
            batch_size = tf.shape(embeddings)[0]
            labels = tf.cast(labels, tf.int32)
            
            shapley_weights = self.calculate_improved_shapley_values(embeddings, labels)
            
            similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True) / self.temperature
            
            labels_expanded = tf.expand_dims(labels, 1)
            positive_mask = tf.cast(tf.equal(labels_expanded, tf.transpose(labels_expanded)), tf.float32)
            positive_mask = positive_mask - tf.eye(batch_size)
            
            exp_similarities = tf.exp(similarity_matrix)
            exp_similarities = exp_similarities * (1.0 - tf.eye(batch_size))
            
            pos_similarities = exp_similarities * positive_mask
            pos_sums = tf.reduce_sum(pos_similarities, axis=1)
            all_sums = tf.reduce_sum(exp_similarities, axis=1)
            
            has_positives = tf.reduce_sum(positive_mask, axis=1) > 0
            losses = -tf.math.log((pos_sums + 1e-8) / (all_sums + 1e-8))
            
            weighted_losses = losses * shapley_weights
            valid_losses = tf.boolean_mask(weighted_losses, has_positives)
            
            def compute_mean_loss():
                return tf.reduce_mean(valid_losses)
            
            def return_zero_loss():
                return tf.constant(0.0)
            
            return tf.cond(
                tf.size(valid_losses) > 0,
                compute_mean_loss,
                return_zero_loss
            )
        except Exception as e:
            tf.print(f"Error in contrastive loss: {e}")
            return tf.constant(0.0)

    def train_step(self, data):
        try:
            x, y = data
            
            teacher_logits = [teacher(x, training=False) for teacher in self.teachers]
            avg_teacher_logits = tf.reduce_mean(tf.stack(teacher_logits, axis=0), axis=0)

            with tf.GradientTape() as tape:
                intermediate_features = self.feature_extractor(x, training=True)
                student_logits = self.student(x, training=True)
                embeddings = self.projection(intermediate_features, training=True)
                
                classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y, student_logits, from_logits=False
                )
                classification_loss = tf.reduce_mean(classification_loss)
                
                contrastive_loss = self.improved_supervised_contrastive_loss(embeddings, y)
                
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(avg_teacher_logits / self.temperature, axis=1),
                    tf.nn.softmax(student_logits / self.temperature, axis=1)
                ) * (self.temperature ** 2)
                
                total_loss = (0.6 * classification_loss + 
                             0.3 * contrastive_loss + 
                             0.1 * distillation_loss)
                
                importance_weight = tf.reduce_mean(self.feature_importance)
                total_loss = total_loss * (0.8 + 0.2 * importance_weight)

            trainable_vars = self.student.trainable_variables + self.projection.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)
            gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
            
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.compiled_metrics.update_state(y, student_logits)
            
            results = {}
            for metric in self.metrics:
                result = metric.result()
                if isinstance(result, dict):
                    result = tf.reduce_mean([tf.cast(v, tf.float32) for v in result.values()])
                results[metric.name] = result

            results.update({
                "classification_loss": classification_loss,
                "contrastive_loss": contrastive_loss,
                "distillation_loss": distillation_loss,
                "total_loss": total_loss,
                "importance_weight": importance_weight
            })
            
            return results
        except Exception as e:
            tf.print(f"Error in train_step: {e}")
            return {"total_loss": tf.constant(0.0)}

    def test_step(self, data):
        try:
            x, y = data
            student_logits = self.student(x, training=False)
            
            classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, student_logits, from_logits=False
            )
            classification_loss = tf.reduce_mean(classification_loss)
            
            standard_predictions = tf.argmax(student_logits, axis=1)
            standard_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y, tf.int32), tf.cast(standard_predictions, tf.int32)), tf.float32))
            
            intermediate_features = self.feature_extractor(x, training=False)
            embeddings = self.projection(intermediate_features, training=False)
            contrastive_loss = self.improved_supervised_contrastive_loss(embeddings, y)
            
            embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
            similarity_matrix = tf.matmul(embeddings_norm, embeddings_norm, transpose_b=True)
            similarity_matrix = similarity_matrix - tf.eye(tf.shape(similarity_matrix)[0]) * 1e9
            most_similar_indices = tf.argmax(similarity_matrix, axis=1)
            most_similar_labels = tf.gather(y, most_similar_indices)
            contrastive_accuracy = tf.reduce_mean(tf.cast(tf.equal(y, most_similar_labels), tf.float32))
            
            self.compiled_metrics.update_state(y, student_logits)
            results = {}
            
            for metric in self.metrics:
                result = metric.result()
                if isinstance(result, dict):
                    result = tf.reduce_mean([tf.cast(v, tf.float32) for v in result.values()])
                results[metric.name] = result
            
            results.update({
                "classification_loss": classification_loss,
                "contrastive_loss": contrastive_loss,
                "contrastive_accuracy": contrastive_accuracy,
                "standard_accuracy": standard_accuracy,
                "test_loss": classification_loss
            })
            
            return results
        except Exception as e:
            tf.print(f"Error in test_step: {e}")
            return {"test_loss": tf.constant(0.0)}

    def call(self, inputs, training=None, mask=None):
        return self.student(inputs, training=training)

##############################################################################
# Enhanced Model Definitions
##############################################################################
def create_student_model():
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

# Add MedMNIST import at the top
import medmnist
from medmnist import INFO, Evaluator

def load_mnist_dataset(image_size=28):
    """Enhanced dataset loading with BreastMNIST for server-side distillation"""
    try:
        # Use BreastMNIST for server-side multi-teacher distillation
        info = INFO['breastmnist']
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        
        DataClass = getattr(medmnist, info['python_class'])
        
        # Load BreastMNIST dataset
        train_dataset = DataClass(split='train', transform=None, download=True, size=image_size)
        test_dataset = DataClass(split='test', transform=None, download=True, size=image_size)
        
        # Convert to numpy arrays
        x_train = train_dataset.imgs.astype("float32") / 255.0
        y_train = train_dataset.labels.flatten()
        x_test = test_dataset.imgs.astype("float32") / 255.0
        y_test = test_dataset.labels.flatten()
        
        # Ensure proper shape for grayscale images
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
        
        print(f"BreastMNIST Dataset loaded: Train shape {x_train.shape}, Test shape {x_test.shape}")
        print(f"Number of classes: {n_classes}, Task: {task}")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        print(f"Error loading BreastMNIST dataset: {e}")
        raise

# Update model creation functions to support different input sizes
# Add the missing export_metrics_to_excel function after the global variables section (around line 40)
def export_metrics_to_excel():
    """Export federated learning metrics to Excel"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive metrics dataframe
        all_metrics = []
        
        # Add Model A metrics
        for i, (round_num, acc, loss) in enumerate(zip(
            epoch_metrics["Model_A"]["rounds"],
            epoch_metrics["Model_A"]["accuracies"], 
            epoch_metrics["Model_A"]["losses"]
        )):
            all_metrics.append({
                'Model': 'Model_A',
                'Round': round_num,
                'Accuracy': acc,
                'Loss': loss
            })
        
        # Add Model B metrics
        for i, (round_num, acc, loss) in enumerate(zip(
            epoch_metrics["Model_B"]["rounds"],
            epoch_metrics["Model_B"]["accuracies"], 
            epoch_metrics["Model_B"]["losses"]
        )):
            all_metrics.append({
                'Model': 'Model_B',
                'Round': round_num,
                'Accuracy': acc,
                'Loss': loss
            })
        
        # Add Model C metrics
        for i, (round_num, acc, loss) in enumerate(zip(
            epoch_metrics["Model_C"]["rounds"],
            epoch_metrics["Model_C"]["accuracies"], 
            epoch_metrics["Model_C"]["losses"]
        )):
            all_metrics.append({
                'Model': 'Model_C',
                'Round': round_num,
                'Accuracy': acc,
                'Loss': loss
            })
        
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            filename = f'federated_learning_metrics_{timestamp}.xlsx'
            df.to_excel(filename, index=False)
            print(f"[Excel Export] Federated learning metrics exported to {filename}")
        else:
            print("[Excel Export] No metrics to export")
            
    except Exception as e:
        print(f"Error exporting federated learning metrics: {e}")

# Fix the create_server_classifier method in the PartitionedFedAvg class (around line 730)
def create_server_classifier(self):
    """Enhanced server classifier for all three models"""
    if self.model_name == "Model_A":
        input_size = 128
    elif self.model_name == "Model_B":
        input_size = 64
    else:  # Model_C
        input_size = 32
        
    return keras.Sequential([
        keras.Input(shape=(input_size,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')  # Changed from 10 to 2 for BreastMNIST
    ])

# Fix Model A creation (around line 350)
def create_model_A(input_shape=(28, 28, 1)):
    feature_extractor = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu")
    ])
    
    server_classifier = keras.Sequential([
        keras.Input(shape=(128,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')  # Changed to 2 classes for BreastMNIST
    ])
    
    full_model = keras.Sequential([feature_extractor, server_classifier])
    return full_model

# Fix Model B creation (around line 380)
def create_model_B(input_shape=(28, 28, 1)):
    feature_extractor = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu")
    ])
    
    server_classifier = keras.Sequential([
        keras.Input(shape=(64,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')  # Changed to 2 classes for BreastMNIST
    ])
    
    full_model = keras.Sequential([feature_extractor, server_classifier])
    return full_model

# Fix Model C creation (around line 410)
def create_model_C(input_shape=(28, 28, 1)):
    """Create ultra-light Model C architecture"""
    feature_extractor = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(8, kernel_size=(5, 5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu")
    ])
    
    server_classifier = keras.Sequential([
        keras.Input(shape=(32,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')  # Changed to 2 classes for BreastMNIST
    ])
    
    full_model = keras.Sequential([feature_extractor, server_classifier])
    return full_model

# Fix student model creation (around line 430)
def create_student_model(input_shape=(28, 28, 1)):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), 
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),  # Changed to 2 classes for BreastMNIST
    ])
    return model

# Update the perform_multi_teacher_distillation function
def perform_multi_teacher_distillation(image_size=28):
    """Enhanced multi-teacher knowledge distillation with BreastMNIST"""
    try:
        print("\n[Distillation] Starting multi-teacher knowledge distillation with BreastMNIST...")
        
        # Load BreastMNIST dataset
        x_train, y_train, x_test, y_test = load_mnist_dataset(image_size=image_size)
        input_shape = (image_size, image_size, 1)
        
        # Load the three teacher models from the final round (round 10)
        print("[Distillation] Loading teacher models...")
        
        # Load Model A (Teacher 1)
        teacher_A = create_model_A(input_shape=input_shape)
        teacher_A = load_npz_weights_into_model(teacher_A, "Model_A-round-10-weights.npz")
        teacher_A.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        # Load Model B (Teacher 2)
        teacher_B = create_model_B(input_shape=input_shape)
        teacher_B = load_npz_weights_into_model(teacher_B, "Model_B-round-10-weights.npz")
        teacher_B.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        # Load Model C (Teacher 3)
        teacher_C = create_model_C(input_shape=input_shape)
        teacher_C = load_npz_weights_into_model(teacher_C, "Model_C-round-10-weights.npz")
        teacher_C.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        teachers = [teacher_A, teacher_B, teacher_C]
        
        # Calculate feature importance using the existing function
        print("[Distillation] Calculating feature importance...")
        try:
            feature_importance = calculate_feature_importance(
                "Model_A_feature_extractor-round-10.h5",
                "Model_B_feature_extractor-round-10.h5", 
                "Model_C_feature_extractor-round-10.h5"
            )
            print(f"[Distillation] Feature importance calculated: {np.mean(feature_importance):.4f}")
        except Exception as e:
            print(f"[Distillation] Error calculating feature importance: {e}")
            feature_importance = np.ones(128)  # Default fallback
        
        # Create student model
        print("[Distillation] Creating student model...")
        student = create_student_model(input_shape=input_shape)
        
        # Initialize distiller
        distiller = MultiTeacherDistiller(
            student=student,
            teachers=teachers,
            feature_importance=feature_importance
        )
        
        # Compile distiller
        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.3,
            temperature=0.1
        )
        
        # Create epoch tracker for Excel export
        epoch_tracker = EnhancedEpochTracker()
        
        print("[Distillation] Starting training...")
        
        # Train the distiller
        history = distiller.fit(
            x_train, y_train,
            epochs=20,
            validation_data=(x_test, y_test),
            batch_size=128,
            callbacks=[epoch_tracker],
            verbose=1
        )
        
        # Export distillation metrics to Excel
        epoch_tracker.export_to_excel()
        
        # Save final student model
        distiller.student.save("final_student_model.h5")
        
        # Evaluate final student model
        final_student_model = keras.models.load_model("final_student_model.h5", compile=False)
        final_student_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        test_loss, test_accuracy = final_student_model.evaluate(x_test, y_test, verbose=1)
        print(f"\n[Distillation] Final Student Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Save final metrics
        with open("final_metrics.txt", "w") as f:
            f.write(f"Final Student Model Test Loss: {test_loss:.4f}\n")
            f.write(f"Final Student Model Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Multi-teacher distillation completed successfully\n")
        
        print("[Distillation] Multi-teacher knowledge distillation completed successfully!")
        
    except Exception as e:
        print(f"[Distillation] Error in multi-teacher distillation: {e}")
        with open("distillation_error.txt", "w") as f:
            f.write(f"Multi-teacher distillation error: {str(e)}\n")
        raise

# Remove the duplicate load_mnist_dataset function (lines 632-647)
# Delete these lines:
##############################################################################
# Enhanced Dataset Loading
##############################################################################
def load_mnist_dataset():
    """Enhanced dataset loading with proper error handling"""
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        print(f"Dataset loaded: Train shape {x_train.shape}, Test shape {x_test.shape}")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

##############################################################################
# Enhanced Utility Functions
##############################################################################
def load_npz_weights_into_model(model, model_weight_path):
    """Enhanced weight loading with error handling"""
    try:
        with np.load(model_weight_path) as data:
            weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        model.set_weights(weights)
        return model
    except Exception as e:
        print(f"Error loading weights from {model_weight_path}: {e}")
        return model

def fit_and_evaluate_student_model(student_model, x_train, y_train, x_test, y_test, epochs=10):
    """Enhanced student model training with epoch tracking"""
    try:
        student_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        class StudentEpochTracker(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                epoch_info = {
                    'epoch': epoch + 1,
                    'loss': logs.get('loss', 0),
                    'accuracy': logs.get('accuracy', 0),
                    'val_loss': logs.get('val_loss', 0),
                    'val_accuracy': logs.get('val_accuracy', 0)
                }
                
                with open("student_epochs.txt", "a") as f:
                    f.write(f"Student Epoch {epoch_info['epoch']}: "
                           f"Loss={epoch_info['loss']:.4f}, "
                           f"Accuracy={epoch_info['accuracy']:.4f}, "
                           f"Val_Loss={epoch_info['val_loss']:.4f}, "
                           f"Val_Accuracy={epoch_info['val_accuracy']:.4f}\n")
                
                print(f"\n[Student] Epoch {epoch_info['epoch']}: "
                      f"Loss={epoch_info['loss']:.4f}, Accuracy={epoch_info['accuracy']:.4f}")
        
        student_epoch_tracker = StudentEpochTracker()
        
        student_model.fit(
            x_train, y_train, 
            epochs=epochs, 
            validation_data=(x_test, y_test),
            callbacks=[student_epoch_tracker],
            verbose=1
        )
        
        test_loss, test_accuracy = student_model.evaluate(x_test, y_test, verbose=0)
        return test_loss, test_accuracy
    except Exception as e:
        print(f"Error in student model training: {e}")
        return 0.0, 0.0

##############################################################################
# Enhanced Aggregation Functions
##############################################################################
def aggregate_fit_metrics(metrics):
    """Enhanced fit metrics aggregation"""
    return {}

def aggregate_metrics(metrics):
    """Enhanced evaluation metrics aggregation"""
    if not metrics:
        return {}
    
    try:
        total_examples = sum([num_examples for num_examples, _ in metrics])
        weighted_accuracy = sum([num_examples * m["accuracy"] for num_examples, m in metrics]) / total_examples
        return {"accuracy": weighted_accuracy}
    except Exception as e:
        print(f"Error in metrics aggregation: {e}")
        return {"accuracy": 0.0}

##############################################################################
# Enhanced Partitioned FedAvg Strategy
##############################################################################
class PartitionedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.server_classifier = self.create_server_classifier()
        self.track_file = "model_track.txt"
        if not os.path.exists(self.track_file):
            with open(self.track_file, "w") as f:
                f.write("")
    
    def create_server_classifier(self):
        """Enhanced server classifier for all three models"""
        if self.model_name == "Model_A":
            input_size = 128
        elif self.model_name == "Model_B":
            input_size = 64
        else:  # Model_C
            input_size = 32
            
        return keras.Sequential([
            keras.Input(shape=(input_size,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
    
    def mark_model_completed(self, model_name):
        """Enhanced model completion tracking"""
        try:
            with open(self.track_file, "r") as f:
                lines = f.readlines()

            model_entry = f"completed_{model_name}==True\n"
            
            if model_entry not in lines:
                with open(self.track_file, "a") as f:
                    f.write(model_entry)
                print(f"[Server] {model_name} training completed and logged.")
        except Exception as e:
            print(f"Error marking model completion: {e}")

    def check_models_completed(self):
        """Enhanced completion checking"""
        try:
            with open(self.track_file, "r") as f:
                lines = f.readlines()
        
            return ("completed_Model_A==True\n" in lines and 
                   "completed_Model_B==True\n" in lines and 
                   "completed_Model_C==True\n" in lines)
        except Exception as e:
            print(f"Error checking model completion: {e}")
            return False
    
    def aggregate_fit(self, rnd, results, failures):
        """Enhanced aggregation with comprehensive tracking"""
        try:
            aggregated_result = super().aggregate_fit(rnd, results, failures)
            
            if aggregated_result is not None:
                aggregated_weights, _ = aggregated_result
                aggregated_weights_numpy = fl.common.parameters_to_ndarrays(aggregated_weights)
                
                # Create and save feature extractor
                feature_extractor = self.create_feature_extractor_model()
                feature_extractor.set_weights(aggregated_weights_numpy)
                feature_extractor.save(f"{self.model_name}_feature_extractor-round-{rnd}.h5")
                
                # Create and save complete server model
                server_model = keras.Sequential([feature_extractor, self.server_classifier])
                server_model.save(f"{self.model_name}_server_model-round-{rnd}.h5")
                
                # Save in old format for compatibility
                np.savez(f"{self.model_name}-round-{rnd}-weights.npz", *aggregated_weights_numpy)
                
                # Enhanced logging
                with open("server_rounds.txt", "a") as f:
                    f.write(f"{self.model_name} - Round {rnd} completed at {tf.timestamp()}\n")
                
                print(f"\n[Server] {self.model_name} - Round {rnd} completed and weights aggregated")
                
                # Track epoch metrics
                epoch_metrics[self.model_name]["server_epochs"].append(rnd)
                
                if rnd == 10:
                    self.mark_model_completed(self.model_name)
                    if self.check_models_completed() and self.model_name == "Model_C":
                        print("\n[Server] All three models completed. Running final distillation...\n")
                        perform_multi_teacher_distillation()
            
            return aggregated_result
        except Exception as e:
            print(f"Error in aggregate_fit: {e}")
            return None
    
    def create_feature_extractor_model(self):
        """Enhanced feature extractor creation for all three models"""
        if self.model_name == "Model_A":
            return keras.Sequential([
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu")
            ])
        elif self.model_name == "Model_B":
            return keras.Sequential([
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu")
            ])
        else:  # Model_C
            return keras.Sequential([
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(8, kernel_size=(5, 5), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(32, activation="relu")
            ])
    
    # Remove the duplicate perform_multi_teacher_distillation function (lines 616-721)
    # Keep only the first definition (lines 508-615)
    
    # Remove the standalone aggregate_evaluate function (lines 722-775)
    # The correct aggregate_evaluate method is already in the PartitionedFedAvg class
    
    # Fix the aggregate_evaluate method in the PartitionedFedAvg class
    def aggregate_evaluate(self, rnd, results, failures):
        """Enhanced evaluation aggregation with comprehensive tracking"""
        try:
            if not results:
                return None, {}
    
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
    
            accuracy = aggregated_metrics.get('accuracy', 0.0)
            print(f"\n[Server] {self.model_name} - Round {rnd} Accuracy: {accuracy:.4f}")
            print(f"[Server] {self.model_name} - Round {rnd} Loss: {aggregated_loss:.4f}")
    
            # Enhanced metrics storage for all three models
            if self.model_name == "Model_A":
                accuraciess_A.append(accuracy)
                losses_A.append(aggregated_loss)
                epoch_metrics["Model_A"]["rounds"].append(rnd)
                epoch_metrics["Model_A"]["accuracies"].append(accuracy)
                epoch_metrics["Model_A"]["losses"].append(aggregated_loss)
            elif self.model_name == "Model_B":
                accuraciess_B.append(accuracy)
                losses_B.append(aggregated_loss)
                epoch_metrics["Model_B"]["rounds"].append(rnd)
                epoch_metrics["Model_B"]["accuracies"].append(accuracy)
                epoch_metrics["Model_B"]["losses"].append(aggregated_loss)
            elif self.model_name == "Model_C":
                accuraciess_C.append(accuracy)
                losses_C.append(aggregated_loss)
                epoch_metrics["Model_C"]["rounds"].append(rnd)
                epoch_metrics["Model_C"]["accuracies"].append(accuracy)
                epoch_metrics["Model_C"]["losses"].append(aggregated_loss)
            
            # Export to Excel after each round
            export_metrics_to_excel()
            
            # Enhanced metrics file writing for all three models
            with open("server_epoch_metrics.txt", "w") as f:
                f.write("=== Enhanced Server Epoch Metrics ===\n\n")
                f.write("Model A Metrics:\n")
                for i, (r, acc, loss) in enumerate(zip(epoch_metrics["Model_A"]["rounds"], 
                                                      epoch_metrics["Model_A"]["accuracies"], 
                                                      epoch_metrics["Model_A"]["losses"])):
                    f.write(f"Round {r}: Accuracy={acc:.4f}, Loss={loss:.4f}\n")
                
                f.write("\nModel B Metrics:\n")
                for i, (r, acc, loss) in enumerate(zip(epoch_metrics["Model_B"]["rounds"], 
                                                      epoch_metrics["Model_B"]["accuracies"], 
                                                      epoch_metrics["Model_B"]["losses"])):
                    f.write(f"Round {r}: Accuracy={acc:.4f}, Loss={loss:.4f}\n")
                
                f.write("\nModel C Metrics:\n")
                for i, (r, acc, loss) in enumerate(zip(epoch_metrics["Model_C"]["rounds"], 
                                                      epoch_metrics["Model_C"]["accuracies"], 
                                                      epoch_metrics["Model_C"]["losses"])):
                    f.write(f"Round {r}: Accuracy={acc:.4f}, Loss={loss:.4f}\n")
                
                f.write(f"\nLast Updated: {tf.timestamp()}\n")
    
            return aggregated_loss, aggregated_metrics
        except Exception as e:
            print(f"Error in aggregate_evaluate: {e}")
            return 0.0, {"accuracy": 0.0}

##############################################################################
# Enhanced Server Startup
##############################################################################
def start_server(port, model_name):
    """Enhanced server startup with error handling"""
    try:
        strategy = PartitionedFedAvg(
            model_name=model_name,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
        )

        print(f"Starting {model_name} FL server on port {port}...")
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}", 
            config=fl.server.ServerConfig(num_rounds=10), 
            strategy=strategy
        )
    except Exception as e:
        print(f"Error starting server for {model_name}: {e}")
        with open(f"server_error_{model_name}.txt", "w") as f:
            f.write(f"Server error for {model_name}: {str(e)}\n")

##############################################################################
# Enhanced Main Execution
##############################################################################
if __name__ == "__main__":
    # Enhanced cleanup
    cleanup_files = [
        "model_track.txt", "model_epoch.txt", "server_rounds.txt", 
        "server_epoch_metrics.txt", "distillation_epochs.txt", 
        "student_epochs.txt", "final_metrics.txt", "distillation_error.txt"
    ]
    
    for file_path in cleanup_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' has been deleted.")
        else:
            print(f"File '{file_path}' does not exist.")

    # Start servers with enhanced error handling for all three models
    try:
        process_A = multiprocessing.Process(target=start_server, args=(8080, "Model_A"))
        process_B = multiprocessing.Process(target=start_server, args=(8081, "Model_B"))
        process_C = multiprocessing.Process(target=start_server, args=(8082, "Model_C"))
        
        process_A.start()
        process_B.start()
        process_C.start()  # Start Model C server
        
        print("All three federated learning servers started successfully.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        with open("main_error.txt", "w") as f:
            f.write(f"Main execution error: {str(e)}\n")