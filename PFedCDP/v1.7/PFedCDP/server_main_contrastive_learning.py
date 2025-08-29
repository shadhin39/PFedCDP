import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import multiprocessing
import os
import h5py
from calculate_feature_importance import calculate_feature_importance
from dataset_loaders import load_mias_dataset
import cv2
# Global tracking variables
accuraciess_A = []
losses_A = []
aggregated_weight_A = None
aggregated_weight_B = None
federated_rounds = 10
accuraciess_B = []
losses_B = []

# Enhanced epoch tracking with detailed metrics
epoch_metrics = {
    "Model_A": {
        "rounds": [], "accuracies": [], "losses": [], 
        "precisions": [], "recalls": [], "f1_scores": [],
        "aucs": [], "specificities": [], "sensitivities": [],
        "client_epochs": [], "server_epochs": []
    },
    "Model_B": {
        "rounds": [], "accuracies": [], "losses": [], 
        "precisions": [], "recalls": [], "f1_scores": [],
        "aucs": [], "specificities": [], "sensitivities": [],
        "client_epochs": [], "server_epochs": []
    },
    "Model_C": {
        "rounds": [], "accuracies": [], "losses": [], 
        "precisions": [], "recalls": [], "f1_scores": [],
        "aucs": [], "specificities": [], "sensitivities": [],
        "client_epochs": [], "server_epochs": []
    }
}
def weighted_average(metrics):
    """Aggregate metrics from multiple clients using weighted average."""
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric names from the first client
    if metrics:
        metric_names = list(metrics[0][1].keys())
        
        for metric_name in metric_names:
            # Calculate weighted average for each metric
            total_examples = sum([num_examples for num_examples, _ in metrics])
            weighted_sum = sum([num_examples * m[metric_name] for num_examples, m in metrics])
            aggregated[metric_name] = weighted_sum / total_examples if total_examples > 0 else 0.0
    
    return aggregated
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
        # Enhanced metrics compilation
        enhanced_metrics = [
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
        if metrics:
            enhanced_metrics.extend(metrics)
        
        super().compile(optimizer=optimizer, metrics=enhanced_metrics)
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
            
            # Get predictions from all three teachers
            teacher_logits = [teacher(x, training=False) for teacher in self.teachers]
            
            # Weighted average of teacher logits (give more weight to larger models)
            weights = [0.4, 0.35, 0.25]  # A=40%, B=35%, C=25%
            weighted_teacher_logits = tf.reduce_sum([
                w * logits for w, logits in zip(weights, teacher_logits)
            ], axis=0)

            with tf.GradientTape() as tape:
                intermediate_features = self.feature_extractor(x, training=True)
                student_logits = self.student(x, training=True)
                embeddings = self.projection(intermediate_features, training=True)
                
                classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y, student_logits, from_logits=False
                )
                classification_loss = tf.reduce_mean(classification_loss)
                
                contrastive_loss = self.improved_supervised_contrastive_loss(embeddings, y)
                
                # Use weighted teacher logits for distillation
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(weighted_teacher_logits / self.temperature, axis=1),
                    tf.nn.softmax(student_logits / self.temperature, axis=1)
                ) * (self.temperature ** 2)
                
                # Adjusted weights for three-teacher setup
                total_loss = (0.5 * classification_loss + 
                             0.3 * contrastive_loss + 
                             0.2 * distillation_loss)
                
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
# Add new imports at the top


def create_student_model():
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2 classes for CBIS-DDSM
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

def create_model_A():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

def create_model_B():
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

def create_model_C():
    model = keras.Sequential([
        layers.Conv2D(8, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

# Update the multi-teacher distillation function
def perform_multi_teacher_distillation():
    """Enhanced distillation with three teachers (A, B, C)"""
    try:
        x_train, y_train, x_test, y_test = load_mnist_dataset()
        
        # Load and compile all three teacher models
        teacher_model_A = keras.models.load_model("Model_A_server_model-round-10.h5", compile=False)
        teacher_model_B = keras.models.load_model("Model_B_server_model-round-10.h5", compile=False)
        teacher_model_C = keras.models.load_model("Model_C_server_model-round-10.h5", compile=False)
        
        for teacher in [teacher_model_A, teacher_model_B, teacher_model_C]:
            teacher.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        teacher_models = [teacher_model_A, teacher_model_B, teacher_model_C]
        
        # Calculate feature importance for all three models
        feature_importance_A = calculate_feature_importance(teacher_model_A, x_train, y_test)
        feature_importance_B = calculate_feature_importance(teacher_model_B, x_train, y_test)
        feature_importance_C = calculate_feature_importance(teacher_model_C, x_train, y_test)
        
        # Average feature importance across all three models
        average_feature_importance = tf.reduce_mean(
            tf.stack([
                tf.convert_to_tensor(feature_importance_A, dtype=tf.float32),
                tf.convert_to_tensor(feature_importance_B, dtype=tf.float32),
                tf.convert_to_tensor(feature_importance_C, dtype=tf.float32)
            ]), 
            axis=0
        )

        # Initialize distiller with three teachers
        distiller = MultiTeacherDistiller(
            student=create_student_model(), 
            teachers=teacher_models,
            feature_importance=average_feature_importance.numpy(),
            embedding_dim=256,
            epsilon=0.01,
            current_epoch=1
        )
        
        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0008, weight_decay=1e-4),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=4.0,
        )
        
        batch_size = 128
        epochs = 15
        
        # Enhanced callbacks
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_standard_accuracy', factor=0.5, patience=3, min_lr=1e-6, mode='max'
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_standard_accuracy', patience=5, restore_best_weights=True, mode='max'
        )
        
        class EnhancedEpochTracker(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                epoch_info = {
                    'epoch': epoch + 1,
                    'loss': logs.get('loss', 0),
                    'accuracy': logs.get('sparse_categorical_accuracy', 0),
                    'val_loss': logs.get('val_loss', 0),
                    'val_accuracy': logs.get('val_sparse_categorical_accuracy', 0)
                }
                
                with open("distillation_epochs.txt", "a") as f:
                    f.write(f"Epoch {epoch_info['epoch']}: Loss={epoch_info['loss']:.4f}, "
                           f"Accuracy={epoch_info['accuracy']:.4f}, "
                           f"Val_Loss={epoch_info['val_loss']:.4f}, "
                           f"Val_Accuracy={epoch_info['val_accuracy']:.4f}\n")
                
                print(f"\n[Distillation] Epoch {epoch_info['epoch']}: "
                      f"Loss={epoch_info['loss']:.4f}, Accuracy={epoch_info['accuracy']:.4f}")
        
        epoch_tracker = EnhancedEpochTracker()
        
        # Train with proper data handling
        distiller.fit(
            x_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[lr_scheduler, early_stopping, epoch_tracker],
            verbose=1
        )

        # Save and evaluate
        distiller.student.save("final_student_model.h5")
        
        final_student_model = keras.models.load_model("final_student_model.h5", compile=False)
        final_student_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        test_loss, test_accuracy = final_student_model.evaluate(x_test, y_test, verbose=1)
        print(f"\nFinal Student Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        with open("final_metrics.txt", "w") as f:
            f.write(f"Final Student Model Test Loss: {test_loss:.4f}\n")
            f.write(f"Final Student Model Test Accuracy: {test_accuracy:.4f}\n")
            
    except Exception as e:
        print(f"Error in three-teacher distillation: {e}")
        with open("distillation_error.txt", "w") as f:
            f.write(f"Three-teacher distillation error: {str(e)}\n")

# Update the MultiTeacherDistiller train_step to handle three teachers
class MultiTeacherDistiller(keras.Model):
    def train_step(self, data):
        try:
            x, y = data
            
            # Get predictions from all three teachers
            teacher_logits = [teacher(x, training=False) for teacher in self.teachers]
            
            # Weighted average of teacher logits (give more weight to larger models)
            weights = [0.4, 0.35, 0.25]  # A=40%, B=35%, C=25%
            weighted_teacher_logits = tf.reduce_sum([
                w * logits for w, logits in zip(weights, teacher_logits)
            ], axis=0)

            with tf.GradientTape() as tape:
                intermediate_features = self.feature_extractor(x, training=True)
                student_logits = self.student(x, training=True)
                embeddings = self.projection(intermediate_features, training=True)
                
                classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y, student_logits, from_logits=False
                )
                classification_loss = tf.reduce_mean(classification_loss)
                
                contrastive_loss = self.improved_supervised_contrastive_loss(embeddings, y)
                
                # Use weighted teacher logits for distillation
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(weighted_teacher_logits / self.temperature, axis=1),
                    tf.nn.softmax(student_logits / self.temperature, axis=1)
                ) * (self.temperature ** 2)
                
                # Adjusted weights for three-teacher setup
                total_loss = (0.5 * classification_loss + 
                             0.3 * contrastive_loss + 
                             0.2 * distillation_loss)
                
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

# Add this function after the imports and before the MultiTeacherDistiller class
def load_breastmnist_dataset():
    """Loads the BreastMNIST dataset for post-hoc scenarios."""
    info = INFO['breastmnist']
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load train and test data
    train_dataset = DataClass(split='train', download=True, size=28)  # 28x28 pixels
    test_dataset = DataClass(split='test', download=True, size=28)
    
    # Extract data and labels
    x_train = train_dataset.imgs
    y_train = train_dataset.labels.flatten()
    x_test = test_dataset.imgs
    y_test = test_dataset.labels.flatten()
    
    # Normalize pixel values (scale to [0,1])
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Ensure proper shape for grayscale images
    if len(x_train.shape) == 3:  # (N, H, W) -> (N, H, W, 1)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    
    return x_train, y_train, x_test, y_test

def load_chestmnist_dataset():
    """Loads the ChestMNIST dataset for server public data in multi-teacher distillation."""
    info = INFO['chestmnist']
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load train and test data
    train_dataset = DataClass(split='train', download=True, size=28)  # 28x28 pixels
    test_dataset = DataClass(split='test', download=True, size=28)
    
    # Extract data and labels
    x_train = train_dataset.imgs
    y_train = train_dataset.labels.flatten()
    x_test = test_dataset.imgs
    y_test = test_dataset.labels.flatten()
    
    # Normalize pixel values (scale to [0,1])
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Ensure proper shape for grayscale images
    if len(x_train.shape) == 3:  # (N, H, W) -> (N, H, W, 1)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    
    return x_train, y_train, x_test, y_test

# Replace the existing dataset loading functions
def load_mias_server_dataset():
    """Loads the MIAS dataset for public server data."""
    return load_mias_dataset()

# Update the load_mnist_dataset function to use MIAS
def load_mnist_dataset():
    """Loads the MIAS dataset for server (replaces previous datasets)."""
    return load_mias_server_dataset()

# Update model architectures for mammography (larger input size)
def create_student_model():
    model = keras.Sequential([
        keras.Input(shape=(224, 224, 1)),  # Updated for mammography
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"), 
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(3, activation="softmax"), # 3 classes for MIAS (Normal, Benign, Malignant)
    ])
    return model

def create_model_A():
    feature_extractor = keras.Sequential([
        keras.Input(shape=(224, 224, 1)),  # Updated for mammography
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation="relu")
    ])
    
    server_classifier = keras.Sequential([
        keras.Input(shape=(512,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax') # 3 classes for MIAS
    ])
    
    model = keras.Sequential([feature_extractor, server_classifier])
    return model

def create_model_B():
    feature_extractor = keras.Sequential([
        keras.Input(shape=(224, 224, 1)),  # Updated for mammography
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu")
    ])
    
    server_classifier = keras.Sequential([
        keras.Input(shape=(256,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax') # 3 classes for MIAS
    ])
    
    model = keras.Sequential([feature_extractor, server_classifier])
    return model

def create_model_C():
    feature_extractor = keras.Sequential([
        keras.Input(shape=(224, 224, 1)),  # Updated for mammography
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu")
    ])
    
    server_classifier = keras.Sequential([
        keras.Input(shape=(128,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(3, activation='softmax') # 3 classes for MIAS
    ])
    
    model = keras.Sequential([feature_extractor, server_classifier])
    return model

# Update the class indices in calculate_improved_shapley_values
class_indices = tf.range(3)  # Changed from 10 to 3 for MIAS classes

##############################################################################
# 3 Aggregate Fit & Evaluation Metrics
##############################################################################
def aggregate_fit_metrics(metrics):
    """Aggregate the fit metrics (accuracy, loss, etc.) across clients."""
    aggregated_metrics = {}
    for metric_tuple in metrics:
        for key, value in metric_tuple[1].items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = []
            aggregated_metrics[key].append(value)
    return {key: np.mean(values) for key, values in aggregated_metrics.items()}

def aggregate_metrics(metrics):
    """Aggregate the evaluation metrics (accuracy, loss, etc.) across clients."""
    aggregated = {}
    for metric_tuple in metrics:
        for key, value in metric_tuple[1].items():
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(value)
    return {key: np.mean(values) for key, values in aggregated.items()}

##############################################################################
# 4 FedAvg Strategy with Knowledge Distillation
##############################################################################
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_name, **kwargs):
        """Initialize the FedAvg strategy with custom model name."""
        super().__init__(**kwargs)
        self.model_name = model_name
        # Ensure model tracking file exists
        self.track_file = "model_track.txt"
        if not os.path.exists(self.track_file):
            with open(self.track_file, "w") as f:
                f.write("")  # Create an empty file
                 
    def mark_model_completed(self, model_name):
        """Write model completion status to model_track.txt."""
        with open(self.track_file, "r") as f:
            lines = f.readlines()

        model_entry = f"completed_{model_name}==True\n"
        
        if model_entry not in lines:  # Avoid duplicate entries
            with open(self.track_file, "a") as f:
                f.write(model_entry)
            print(f"[Server] {model_name} training completed and logged.")

    def check_models_completed(self):
        """Check if all three models (A, B, C) have completed training."""
        with open(self.track_file, "r") as f:
            lines = f.readlines()

        return ("completed_Model_A==True\n" in lines and 
                "completed_Model_B==True\n" in lines and 
                "completed_Model_C==True\n" in lines)

    def aggregate_fit(self, rnd, results, failures):
        """Aggregate the weights after local training and store them as .h5 files."""
        aggregated_result = super().aggregate_fit(rnd, results, failures)

        if aggregated_result is not None:
            aggregated_weights, _ = aggregated_result  # Extract `parameters` (weights)

            print(f"Saving {self.model_name} round {rnd} weights as .h5...")
            print(f"Saving {self.model_name} round {rnd} weights as .npz...")
            np.savez(f"{self.model_name}-round-{rnd}-weights.npz", *aggregated_result)
            # Convert `Parameters` object to a list of NumPy arrays
            aggregated_weights_numpy = fl.common.parameters_to_ndarrays(aggregated_weights)

            # Create a model instance to load weights
            if self.model_name == "Model_A":
                model = create_model_A()
            elif self.model_name == "Model_B":
                model = create_model_B()
            elif self.model_name == "Model_C":
                model = create_model_C()

            # Assign weights to the model
            model.set_weights(aggregated_weights_numpy)  # Ensure it's a list of NumPy arrays
            
            # Save the model as .h5
            model.save(f"{self.model_name}_server_model-round-{rnd}.h5")

            # Check model completion status
            if rnd == 10:
                self.mark_model_completed(self.model_name)

                # Check if all three models are completed
                if self.check_models_completed():
                    print("\n[Server] All three models completed. Running final distillation...\n")
                    perform_multi_teacher_distillation()

        return aggregated_result
    
    def aggregate_evaluate(self, rnd, results, failures):
        """Aggregate the evaluation results and store comprehensive metrics."""
        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        # Print comprehensive metrics
        print(f"\n{self.model_name} - Round {rnd} Metrics:")
        print(f"  Accuracy: {aggregated_metrics.get('accuracy', 0.0):.4f}")
        print(f"  Loss: {aggregated_loss:.4f}")
        print(f"  Precision: {aggregated_metrics.get('precision', 0.0):.4f}")
        print(f"  Recall: {aggregated_metrics.get('recall', 0.0):.4f}")
        print(f"  F1 Score: {aggregated_metrics.get('f1_score', 0.0):.4f}")
        print(f"  AUC: {aggregated_metrics.get('auc', 0.0):.4f}")
        print(f"  Specificity: {aggregated_metrics.get('specificity', 0.0):.4f}")
        print(f"  Sensitivity: {aggregated_metrics.get('sensitivity', 0.0):.4f}")

        # Store comprehensive metrics in epoch_metrics
        if self.model_name in epoch_metrics:
            epoch_metrics[self.model_name]["rounds"].append(rnd)
            epoch_metrics[self.model_name]["accuracies"].append(aggregated_metrics.get("accuracy", 0.0))
            epoch_metrics[self.model_name]["losses"].append(aggregated_loss)
            epoch_metrics[self.model_name]["precisions"].append(aggregated_metrics.get("precision", 0.0))
            epoch_metrics[self.model_name]["recalls"].append(aggregated_metrics.get("recall", 0.0))
            epoch_metrics[self.model_name]["f1_scores"].append(aggregated_metrics.get("f1_score", 0.0))
            epoch_metrics[self.model_name]["aucs"].append(aggregated_metrics.get("auc", 0.0))
            epoch_metrics[self.model_name]["specificities"].append(aggregated_metrics.get("specificity", 0.0))
            epoch_metrics[self.model_name]["sensitivities"].append(aggregated_metrics.get("sensitivity", 0.0))

        # Store legacy tracking for Model_A and Model_B
        if self.model_name == "Model_A":
            accuraciess_A.append(aggregated_metrics.get("accuracy", 0.0))
            losses_A.append(aggregated_loss)
        elif self.model_name == "Model_B":
            accuraciess_B.append(aggregated_metrics.get("accuracy", 0.0))
            losses_B.append(aggregated_loss)

        # Save comprehensive metrics to file
        with open(f"server_rounds_{self.model_name}.txt", "a") as f:
            f.write(f"Round {rnd}: Accuracy={aggregated_metrics.get('accuracy', 0.0):.4f}, "
                   f"Loss={aggregated_loss:.4f}, Precision={aggregated_metrics.get('precision', 0.0):.4f}, "
                   f"Recall={aggregated_metrics.get('recall', 0.0):.4f}, F1={aggregated_metrics.get('f1_score', 0.0):.4f}, "
                   f"AUC={aggregated_metrics.get('auc', 0.0):.4f}, Specificity={aggregated_metrics.get('specificity', 0.0):.4f}, "
                   f"Sensitivity={aggregated_metrics.get('sensitivity', 0.0):.4f}\n")

        return aggregated_loss, aggregated_metrics

##############################################################################
# 6 Start Federated Learning Server
##############################################################################
def start_server(port, model_name):
    """Start the federated learning server for a given model."""
    strategy = SaveModelStrategy(
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

##############################################################################
# 7 Running FL Servers and Multi-Teacher Distillation
##############################################################################
if __name__ == "__main__":
    
    # Check if the file exists before deleting
    file_path = "model_track.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"File '{file_path}' does not exist.")

    # Check if the file exists before deleting
    file_path1 = "model_epoch.txt"
    if os.path.exists(file_path1):
        os.remove(file_path1)
        print(f"File '{file_path1}' has been deleted.")
    else:
        print(f"File '{file_path1}' does not exist.")

    # Clean up previous server round files
    for model in ["Model_A", "Model_B", "Model_C"]:
        server_file = f"server_rounds_{model}.txt"
        if os.path.exists(server_file):
            os.remove(server_file)
            print(f"File '{server_file}' has been deleted.")

    # Start all three federated learning servers
    process_A = multiprocessing.Process(target=start_server, args=(8080, "Model_A"))
    process_B = multiprocessing.Process(target=start_server, args=(8081, "Model_B"))
    process_C = multiprocessing.Process(target=start_server, args=(8082, "Model_C"))

    process_A.start()
    process_B.start()
    process_C.start()

    # Wait for all processes to complete
    process_A.join()
    process_B.join()
    process_C.join()

    print("\n[Server] All federated learning processes completed.")
