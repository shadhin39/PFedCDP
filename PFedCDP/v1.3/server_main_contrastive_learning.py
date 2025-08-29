import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import multiprocessing
import os
import h5py
from calculate_feature_importance import calculate_feature_importance

# Lists to store accuracies, losses, and aggregated weights over rounds
accuraciess_A = []
losses_A = []
aggregated_weight_A = None
aggregated_weight_B = None
federated_rounds = 10
accuraciess_B = []
losses_B = []

##############################################################################
# Multi-Teacher Distillation Model
#############################################################################
class MultiTeacherDistiller(keras.Model):
    def __init__(self, student, teachers, feature_importance, embedding_dim=256, epsilon=0.01, current_epoch=1):
        super().__init__()
        self.teachers = teachers
        self.student = student
        self.feature_importance = feature_importance
        self.epsilon = epsilon
        self.current_epoch = current_epoch
        self.temperature = 0.1
        
        # Build the student model first to define its input
        dummy_input = tf.zeros((1, 28, 28, 1))
        _ = self.student(dummy_input)
        
        # Improved projection head for contrastive learning
        self.projection = keras.Sequential([
            layers.Dense(embedding_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(embedding_dim // 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(embedding_dim // 4)  # Final embedding without activation
        ])
        
        # Create feature extractor by creating a new model with same layers except the last one
        # This avoids the input access issue with Sequential models
        feature_layers = self.student.layers[:-1]  # All layers except the last (softmax) layer
        self.feature_extractor = keras.Sequential(feature_layers)

    def compile(self, optimizer, metrics=None, distillation_loss_fn=None, alpha=0.3, temperature=0.1):
        """Compile the distiller with required parameters"""
        super().compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def calculate_improved_shapley_values(self, embeddings, labels):
        """Calculate improved Shapley values for contrastive learning"""
        # Normalize embeddings
        embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
        batch_size = tf.shape(embeddings)[0]
        
        # Convert labels to consistent dtype
        labels = tf.cast(labels, tf.int32)
        shapley_values = tf.zeros(batch_size, dtype=tf.float32)
        
        # Process each class using tf.foldl for proper graph execution
        class_indices = tf.range(10, dtype=tf.int32)  # MNIST has 10 classes
        
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
        
        # Normalize to sum to 1
        shapley_values = tf.nn.softmax(shapley_values)
        return shapley_values

    def improved_supervised_contrastive_loss(self, embeddings, labels):
        """Improved supervised contrastive loss with proper InfoNCE formulation"""
        # Normalize embeddings
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        batch_size = tf.shape(embeddings)[0]
        
        # Ensure labels are consistent dtype
        labels = tf.cast(labels, tf.int32)
        
        # Calculate Shapley weights
        shapley_weights = self.calculate_improved_shapley_values(embeddings, labels)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True) / self.temperature
        
        # Create positive mask (same class, excluding self)
        labels_expanded = tf.expand_dims(labels, 1)
        positive_mask = tf.cast(tf.equal(labels_expanded, tf.transpose(labels_expanded)), tf.float32)
        positive_mask = positive_mask - tf.eye(batch_size)  # Remove diagonal
        
        # Apply temperature and compute exponentials
        exp_similarities = tf.exp(similarity_matrix)
        
        # Mask out diagonal elements
        exp_similarities = exp_similarities * (1.0 - tf.eye(batch_size))
        
        # Vectorized computation for better performance
        # Sum positive similarities for each sample
        pos_similarities = exp_similarities * positive_mask
        pos_sums = tf.reduce_sum(pos_similarities, axis=1)
        
        # Sum all similarities (excluding self) for each sample
        all_sums = tf.reduce_sum(exp_similarities, axis=1)
        
        # Compute loss for samples that have positive pairs
        has_positives = tf.reduce_sum(positive_mask, axis=1) > 0
        
        # Compute contrastive loss
        losses = -tf.math.log((pos_sums + 1e-8) / (all_sums + 1e-8))
        
        # Apply Shapley weights and mask for samples with positive pairs
        weighted_losses = losses * shapley_weights
        valid_losses = tf.boolean_mask(weighted_losses, has_positives)
        
        # Use tf.cond instead of Python if statement
        def compute_mean_loss():
            return tf.reduce_mean(valid_losses)
        
        def return_zero_loss():
            return tf.constant(0.0)
        
        return tf.cond(
            tf.size(valid_losses) > 0,
            compute_mean_loss,
            return_zero_loss
        )

    def train_step(self, data):
        x, y = data
        
        # Get teacher predictions
        teacher_logits = [teacher(x, training=False) for teacher in self.teachers]
        avg_teacher_logits = tf.reduce_mean(tf.stack(teacher_logits, axis=0), axis=0)

        with tf.GradientTape() as tape:
            # Get student intermediate features (before softmax)
            intermediate_features = self.feature_extractor(x, training=True)
            
            # Get final student predictions
            student_logits = self.student(x, training=True)
            
            # Project intermediate features for contrastive learning
            embeddings = self.projection(intermediate_features, training=True)
            
            # Calculate classification loss (primary)
            classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
                y, student_logits, from_logits=False
            )
            classification_loss = tf.reduce_mean(classification_loss)
            
            # Calculate contrastive loss (secondary)
            contrastive_loss = self.improved_supervised_contrastive_loss(embeddings, y)
            
            # Calculate distillation loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(avg_teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_logits / self.temperature, axis=1)
            ) * (self.temperature ** 2)
            
            # Combine losses with proper weighting
            # Classification loss should be primary (0.6), contrastive secondary (0.3), distillation tertiary (0.1)
            total_loss = (0.6 * classification_loss + 
                         0.3 * contrastive_loss + 
                         0.1 * distillation_loss)
            
            # Apply feature importance scaling
            importance_weight = tf.reduce_mean(self.feature_importance)
            total_loss = total_loss * (0.8 + 0.2 * importance_weight)

        # Update model
        trainable_vars = self.student.trainable_variables + self.projection.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Gradient clipping for stability
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, student_logits)
        results = {}
        
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                result = tf.reduce_mean([tf.cast(v, tf.float32) for v in result.values()])
            results[metric.name] = result

        # Add loss values
        results.update({
            "classification_loss": classification_loss,
            "contrastive_loss": contrastive_loss,
            "distillation_loss": distillation_loss,
            "total_loss": total_loss,
            "importance_weight": importance_weight
        })
        
        return results

    def test_step(self, data):
        x, y = data
        
        # Get final student predictions
        student_logits = self.student(x, training=False)
        
        # Calculate standard classification loss and accuracy
        classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y, student_logits, from_logits=False
        )
        classification_loss = tf.reduce_mean(classification_loss)
        
        # Calculate standard accuracy
        standard_predictions = tf.argmax(student_logits, axis=1)
        standard_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y, tf.int32), tf.cast(standard_predictions, tf.int32)), tf.float32))
        
        # Optional: Calculate contrastive metrics for monitoring
        intermediate_features = self.feature_extractor(x, training=False)
        embeddings = self.projection(intermediate_features, training=False)
        contrastive_loss = self.improved_supervised_contrastive_loss(embeddings, y)
        
        # Calculate contrastive-based accuracy
        embeddings_norm = tf.nn.l2_normalize(embeddings, axis=1)
        similarity_matrix = tf.matmul(embeddings_norm, embeddings_norm, transpose_b=True)
        similarity_matrix = similarity_matrix - tf.eye(tf.shape(similarity_matrix)[0]) * 1e9
        most_similar_indices = tf.argmax(similarity_matrix, axis=1)
        most_similar_labels = tf.gather(y, most_similar_indices)
        contrastive_accuracy = tf.reduce_mean(tf.cast(tf.equal(y, most_similar_labels), tf.float32))
        
        # Update standard metrics for compatibility
        self.compiled_metrics.update_state(y, student_logits)
        results = {}
        
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                result = tf.reduce_mean([tf.cast(v, tf.float32) for v in result.values()])
            results[metric.name] = result
        
        # Add metrics - use classification loss as primary test loss
        results.update({
            "classification_loss": classification_loss,
            "contrastive_loss": contrastive_loss,
            "contrastive_accuracy": contrastive_accuracy,
            "standard_accuracy": standard_accuracy,
            "test_loss": classification_loss  # Use classification loss as primary test loss
        })
        
        return results

    def call(self, inputs, training=None, mask=None):
        return self.student(inputs, training=training)

##############################################################################
# Student Model Definition
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


##############################################################################
# Multi-teacher distillation weights for the final student model
##############################################################################
# Define Model A with enhanced server classifier
def create_model_A():
    # Feature Extractor (same as client)
    feature_extractor = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu")
    ])
    
    # Enhanced Server Classifier
    server_classifier = keras.Sequential([
        keras.Input(shape=(128,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    # Complete server model
    model = keras.Sequential([feature_extractor, server_classifier])
    return model

# Define Model B with enhanced server classifier
def create_model_B():
    # Feature Extractor (same as client)
    feature_extractor = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu")
    ])
    
    # Enhanced Server Classifier
    server_classifier = keras.Sequential([
        keras.Input(shape=(64,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    # Complete server model
    model = keras.Sequential([feature_extractor, server_classifier])
    return model

##############################################################################
# Multi-Teacher Knowledge Distillation Execution
##############################################################################
def perform_multi_teacher_distillation():
    """Updated distillation using server models with enhanced classifiers"""
    # Load dataset
    x_train, y_train, x_test, y_test = load_mnist_dataset()
    
    # Load server models (feature extractor + enhanced classifier)
    teacher_model_A = keras.models.load_model("Model_A_server_model-round-10.h5", compile=False)
    teacher_model_B = keras.models.load_model("Model_B_server_model-round-10.h5", compile=False)
    
    # Compile teacher models
    teacher_model_A.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    teacher_model_B.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    teacher_models = [teacher_model_A, teacher_model_B]
    
    # Calculate feature importance using the enhanced server models
    feature_importance_A = calculate_feature_importance(teacher_model_A, x_train, y_test)
    feature_importance_B = calculate_feature_importance(teacher_model_B, x_train, y_test)
    average_feature_importance = tf.reduce_mean(
        tf.stack([
            tf.convert_to_tensor(feature_importance_A, dtype=tf.float32),
            tf.convert_to_tensor(feature_importance_B, dtype=tf.float32)
        ]), 
        axis=0
    )

    # Initialize distiller with improved parameters
    distiller = MultiTeacherDistiller(
        student=create_student_model(), 
        teachers=teacher_models,
        feature_importance=average_feature_importance.numpy(),
        embedding_dim=256,  # Increased embedding dimension
        epsilon=0.01,
        current_epoch=1
    )
    
    # Compile distiller with improved parameters
    distiller.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, weight_decay=1e-4),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,  # Reduced alpha for less distillation weight
        temperature=4.0,  # Higher temperature for softer distributions
    )
    
    # Train the model with learning rate scheduling
    batch_size = 128  # Larger batch size for better gradient estimates
    epochs = 15  # More epochs for better convergence
    
    # Learning rate scheduler
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_standard_accuracy', factor=0.5, patience=3, min_lr=1e-6, mode='max'
    )
    
    # Early stopping based on validation accuracy
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_standard_accuracy', patience=5, restore_best_weights=True, mode='max'
    )
    
    distiller.fit(
        x_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[lr_scheduler, early_stopping],
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

##############################################################################
# Load MNIST Dataset
##############################################################################
def load_mnist_dataset():
    """Loads the MNIST dataset and preprocesses it for training and distillation."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values (scale to [0,1])
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand dimensions to match (28,28,1) format (grayscale)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test

##############################################################################
# 2 Utility Functions for Model Weights
##############################################################################
def load_npz_weights_into_model(model, model_weight_path):
    """
    Load weights from .npz file into a Keras/TF2 model.
    
    Args:
        model: The compiled Keras/TF2 model to load weights into.
        model_weight_path: Path to the .npz file containing the weights.
        
    Returns:
        A compiled model with the loaded weights.
    """
    # Load the .npz file
    npz_file = os.path.expanduser(model_weight_path)
    if not os.path.exists(npz_file):
        raise ValueError(f"The model weight file at {model_weight_path} was not found.")
    
    # Load the contents of the .npz file
    weights = np.load(npz_file, allow_pickle=True)
    
    # Extract all layer names from the saved model (assuming they are saved as h5 files)
    # This will help verify that the layers match those in the .npz file.
    saved_layer_names = [layer.name for layer in model.layers]
    
    # Match each layer's name to its corresponding weight array
    for layer_name, weights_dict in zip(saved_layer_names, weights.keys()):
        if 'kernel' in weights_dict and hasattr(model.get_layer(layer_name), 'kernel'):
            # Assuming the kernel is a weight tensor
            weights_array = weights[weights_dict]
            layer = model.get_layer(layer_name)
            layer.kernel.numpy()[:] = np.array(weights_array)
        
        elif 'bias' in weights_dict and hasattr(model.get_layer(layer_name), 'bias'):
            # Assuming bias exists (e.g., Conv2D layers with biases)
            weights_array = weights[weights_dict]
            layer = model.get_layer(layer_name)
            if hasattr(layer, 'bias') and isinstance(layer.bias, tf.Variable):
                layer.bias.numpy()[:] = np.array(weights_array)
        
    
    return model

##############################################################################
# Student Model Evaluation
#############################################################################

def fit_student_model(student_model, train_dataset):
    """Train the student model on the MNIST training dataset."""
    student_model.compile(
        optimizer="sgd",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    history = student_model.fit(train_dataset, epochs=5, verbose=0)
    
    # Extract final training loss and accuracy
    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]
    
    return train_loss, train_acc

def evaluate_student_model(student_model, test_dataset):
    student_model.compile(
        optimizer="sgd",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )   
    loss, accuracy = student_model.evaluate(test_dataset, verbose=0)
    return loss, accuracy

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
        """Check if both Model_A and Model_B have completed training."""
        with open(self.track_file, "r") as f:
            lines = f.readlines()
    
        return "completed_Model_A==True\n" in lines and "completed_Model_B==True\n" in lines

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
            if self.model_name == "Model_B":
                model = create_model_B()

            # Assign weights to the model
            model.set_weights(aggregated_weights_numpy)  # Ensure it's a list of NumPy arrays
            
            # Save the model as .h5
            model.save(f"{self.model_name}-round-{rnd}-weights.h5")

            # Check model completion status
            if rnd == 10:
                self.mark_model_completed(self.model_name)

                # Check if both Model_A and Model_B are completed
                if self.check_models_completed():
                    print("\n[Server] Both models completed. Running final distillation...\n")
                    perform_multi_teacher_distillation()

        return aggregated_result
    
    def aggregate_evaluate(self, rnd, results, failures):
        """Aggregate the evaluation results and store metrics."""
        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)

        print(f"{self.model_name} - Round {rnd} Accuracy: {aggregated_metrics.get('accuracy', 0.0)}")
        print(f"{self.model_name} - Round {rnd} Loss: {aggregated_loss}")

        # Store accuracy and loss separately
        if self.model_name == "Model_A":
            accuraciess_A.append(aggregated_metrics.get("accuracy", 0.0))
            losses_A.append(aggregated_loss)

        elif self.model_name == "Model_B":
            accuraciess_B.append(aggregated_metrics.get("accuracy", 0.0))
            losses_B.append(aggregated_loss)

        return aggregated_loss, aggregated_metrics

##############################################################################
# 6 Start Federated Learning Server
##############################################################################
def start_server(port, model_name):
    """Start the federated learning server for a given model."""
    # Use PartitionedFedAvg instead of SaveModelStrategy
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

    # In the main execution block
    process_A = multiprocessing.Process(target=start_server, args=(8080, "Model_A"))
    process_B = multiprocessing.Process(target=start_server, args=(8081, "Model_B"))
    
    process_A.start()
    process_B.start()

class PartitionedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # Server maintains its own enhanced classifier
        self.server_classifier = self.create_server_classifier()
        # Ensure model tracking file exists
        self.track_file = "model_track.txt"
        if not os.path.exists(self.track_file):
            with open(self.track_file, "w") as f:
                f.write("")  # Create an empty file
    
    def create_server_classifier(self):
        """Enhanced server classifier for multi-teacher distillation"""
        if self.model_name == "Model_A":
            input_size = 128  # Model A feature extractor output
        else:  # Model_B
            input_size = 64   # Model B feature extractor output
            
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
        """Write model completion status to model_track.txt."""
        with open(self.track_file, "r") as f:
            lines = f.readlines()

        model_entry = f"completed_{model_name}==True\n"
        
        if model_entry not in lines:  # Avoid duplicate entries
            with open(self.track_file, "a") as f:
                f.write(model_entry)
            print(f"[Server] {model_name} training completed and logged.")

    def check_models_completed(self):
        """Check if both Model_A and Model_B have completed training."""
        with open(self.track_file, "r") as f:
            lines = f.readlines()
    
        return "completed_Model_A==True\n" in lines and "completed_Model_B==True\n" in lines
    
    def aggregate_fit(self, rnd, results, failures):
        """Aggregate only feature extractor weights"""
        # Standard FedAvg for feature extractors
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_result is not None:
            # Save aggregated feature extractor weights
            aggregated_weights, _ = aggregated_result
            aggregated_weights_numpy = fl.common.parameters_to_ndarrays(aggregated_weights)
            
            # Create feature extractor model and save
            feature_extractor = self.create_feature_extractor_model()
            feature_extractor.set_weights(aggregated_weights_numpy)
            feature_extractor.save(f"{self.model_name}_feature_extractor-round-{rnd}.h5")
            
            # Create complete server model (feature extractor + server classifier)
            server_model = keras.Sequential([feature_extractor, self.server_classifier])
            server_model.save(f"{self.model_name}_server_model-round-{rnd}.h5")
            
            # Also save in the old format for compatibility
            np.savez(f"{self.model_name}-round-{rnd}-weights.npz", *aggregated_weights_numpy)
            
            # Check model completion status
            if rnd == 10:
                self.mark_model_completed(self.model_name)

                # Check if both Model_A and Model_B are completed
                if self.check_models_completed():
                    print("\n[Server] Both models completed. Running final distillation...\n")
                    perform_multi_teacher_distillation()
        
        return aggregated_result
    
    def create_feature_extractor_model(self):
        """Create feature extractor based on model type"""
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
        else:  # Model_B
            return keras.Sequential([
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation="relu")
            ])