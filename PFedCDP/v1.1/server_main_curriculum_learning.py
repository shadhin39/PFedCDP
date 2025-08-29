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
    def __init__(self, student, teachers, feature_importance, epsilon=0.01,current_epoch=1):
        super().__init__()
        self.teachers = teachers
        self.student = student
        self.student_loss_fn = None
        self.distillation_loss_fn = None
        self.alpha = 0.1
        # self.alpha = None
        # self.temperature = None
        self.temperature = 3
        # self.optimizer = None
        self.optimizer = keras.optimizers.Adam()
        self.feature_importance = feature_importance
        self.epsilon = epsilon  
        self.current_epoch = current_epoch  # Track epoch for curriculum learning
        self.feature_importance = tf.convert_to_tensor(self.feature_importance, dtype=tf.float32)
        # self.feature_importance = tf.reshape(self.feature_importance, (-1,))
    def compile(self, optimizer, metrics=None, student_loss_fn=None, distillation_loss_fn=None, alpha=0.1, temperature=3):
        super().compile(optimizer=optimizer, metrics=metrics)  
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.optimizer = optimizer


    def train_step(self, data):
        x, y = data
        epoch_file = "model_epoch.txt"
        
        # Initialize or read current epoch
        if os.path.exists(epoch_file):
            with open(epoch_file, "r") as f:
                current_epoch = int(f.read().strip())
        else:
            current_epoch = 0
            
        teacher_logits = [teacher(x, training=False) for teacher in self.teachers]
        avg_teacher_logits = tf.reduce_mean(tf.stack(teacher_logits, axis=0), axis=0)  # shape: (batch, 10)

        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)  # shape: (batch, 28, 28, 10)

            # Apply feature importance
            student_predictions_weighted = student_predictions * self.feature_importance

            # Reduce spatial dimensions to (batch, num_classes)
            student_predictions_pooled = tf.reduce_mean(student_predictions_weighted, axis=[1, 2])  # (batch, 10)

            # Dynamically match shapes explicitly:
            batch_size = tf.shape(student_predictions_pooled)[0] # get batch size
            y = y[:batch_size]  # slice labels to match predictions explicitly

            num_classes = tf.shape(student_predictions_pooled)[-1]
            y_one_hot = tf.one_hot(y, depth=num_classes)

            # Now compute categorical cross-entropy correctly
            sample_losses = tf.keras.losses.categorical_crossentropy(
                y_one_hot, student_predictions_pooled, from_logits=True
            )

            lambda_t = (1 + self.epsilon) ** current_epoch
            learning_weights = tf.exp(-tf.square(sample_losses))
            learning_weights = tf.where(sample_losses <= lambda_t, 1.0, learning_weights)

            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(avg_teacher_logits[:batch_size] / self.temperature, axis=1),  # match teacher logits batch too
                tf.nn.softmax(student_predictions_pooled / self.temperature, axis=1),
            ) * self.temperature**2

            loss = self.alpha * sample_losses + (1 - self.alpha) * distillation_loss
            loss = tf.reduce_mean(loss * learning_weights)

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, student_predictions_pooled)
        results = {m.name: m.result() for m in self.metrics}

        for key, value in results.items():
            if isinstance(value, dict):
                results[key] = value.get('value', tf.reduce_mean(list(value.values())))

        results.update({
            "loss": loss,
            "student_loss": tf.reduce_mean(sample_losses),
            "distillation_loss": distillation_loss,
            "lambda_t": lambda_t,
            "epoch": current_epoch
        })

        # Increment epoch count and save to file
        current_epoch += 1
        with open(epoch_file, "w") as f:
            f.write(str(current_epoch))


        return results



    # def train_step(self, data):
    #     x, y = data

    #     teacher_logits = [teacher(x, training=False) for teacher in self.teachers]
    #     avg_teacher_logits = tf.reduce_mean(tf.stack(teacher_logits, axis=0), axis=0)  # shape: (32, 10)

    #     with tf.GradientTape() as tape:
    #         student_predictions = self.student(x, training=True)  # shape: (32, 28, 28, 10)

    #         # Apply feature importance (ensure shape matches or is broadcastable)
    #         student_predictions_weighted = student_predictions * self.feature_importance

    #         # Reduce spatial dimensions for student predictions only
    #         student_predictions_pooled = tf.reduce_mean(student_predictions_weighted, axis=[1, 2])  # shape: (32, 10)

    #         num_classes = tf.shape(student_predictions_pooled)[-1]
    #         y_one_hot = tf.one_hot(y, depth=num_classes)

    #         # Compute categorical cross-entropy correctly
    #         sample_losses = tf.keras.losses.categorical_crossentropy(
    #             y_one_hot, student_predictions_pooled, from_logits=True
    #         )

    #         # Compute learning weights v_i based on difficulty
    #         lambda_t = (1 + self.epsilon) ** self.current_epoch  # Dynamic threshold
    #         learning_weights = tf.exp(-tf.square(sample_losses))
    #         learning_weights = tf.where(
    #             sample_losses <= lambda_t,  # If sample loss is less than or equal to threshold
    #             1.0,                        # assign weight = 1.0
    #             learning_weights            # else, use the computed exponential weight
    #         )
            
    #         # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
    #         # The magnitudes of the gradients produced by the soft targets scale
    #         # as 1/T^2, multiply them by T^2 when using both hard and soft targets.

    #         # Compute distillation loss (soft targets from teachers)
    #         distillation_loss = self.distillation_loss_fn(
    #             tf.nn.softmax(avg_teacher_logits / self.temperature, axis=1),
    #             tf.nn.softmax(student_predictions / self.temperature, axis=1),
    #         ) * self.temperature**2

    #         # Compute final weighted loss using curriculum learning
    #         loss = self.alpha * sample_losses + (1 - self.alpha) * distillation_loss
    #         loss = tf.reduce_mean(loss * learning_weights)        # Compute gradients and update student model
    #     trainable_vars = self.student.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #     # Update training metrics
    #     self.compiled_metrics.update_state(y, student_predictions)
    #     results = {m.name: m.result() for m in self.metrics}

    #     # Handle metric dictionary results
    #     for key, value in results.items():
    #         if isinstance(value, dict):
    #             results[key] = value.get('value', tf.reduce_mean(list(value.values())))

    #     # Update custom metrics with student loss and distillation loss
    #     results.update({
    #         "student_loss": tf.reduce_mean(sample_losses),
    #         "distillation_loss": distillation_loss,
    #         "lambda_t": lambda_t  # Log dynamic threshold progression
    #     })

    #     # Increment epoch count for threshold adjustment
    #     self.current_epoch += 1  

    #     return results
    
    # def train_step(self, data):
    #     import os
        
    #     x, y = data
        
    #     epoch_file = "model_track.txt"
        
    #     # Initialize or read current epoch
    #     if os.path.exists(epoch_file):
    #         with open(epoch_file, "r") as f:
    #             current_epoch = int(f.read().strip())
    #     else:
    #         current_epoch = 0
        
    #     # Get teacher predictions (logits) and compute average teacher logits
    #     teacher_logits = [teacher(x, training=False) for teacher in self.teachers]
    #     avg_teacher_logits = tf.reduce_mean(tf.stack(teacher_logits, axis=0), axis=0)

    #     with tf.GradientTape() as tape:
    #         # Student predictions
    #         student_predictions = self.student(x, training=True)
    #         num_classes = tf.shape(student_predictions)[-1]  
    #         y_one_hot = tf.one_hot(y, depth=num_classes)

    #         # Compute per-sample student loss (difficulty score γ_i)
    #         sample_losses = tf.keras.losses.categorical_crossentropy(y_one_hot, student_predictions, from_logits=True)

    #         # Compute learning weights v_i based on difficulty
    #         lambda_t = (1 + self.epsilon) ** current_epoch  
    #         learning_weights = tf.exp(-sample_losses)
    #         learning_weights = tf.where(sample_losses <= lambda_t, 1.0, learning_weights)

    #         # Compute distillation loss (soft targets from teachers)
    #         distillation_loss = self.distillation_loss_fn(
    #             tf.nn.softmax(avg_teacher_logits / self.temperature, axis=1),
    #             tf.nn.softmax(student_predictions / self.temperature, axis=1),
    #         ) * self.temperature**2

    #         # Compute final weighted loss using curriculum learning
    #         loss = self.alpha * sample_losses + (1 - self.alpha) * distillation_loss
    #         loss = tf.reduce_mean(loss * learning_weights)

    #     # Compute gradients and update student model
    #     trainable_vars = self.student.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #     # Update training metrics
    #     self.compiled_metrics.update_state(y, student_predictions)
    #     results = {m.name: m.result() for m in self.metrics}

    #     # Handle metric dictionary results
    #     for key, value in results.items():
    #         if isinstance(value, dict):
    #             results[key] = value.get('value', tf.reduce_mean(list(value.values())))

    #     # Update custom metrics
    #     results.update({
    #         "loss": loss,
    #         "student_loss": tf.reduce_mean(sample_losses),
    #         "distillation_loss": distillation_loss,
    #         "lambda_t": lambda_t,
    #         "epoch": current_epoch
    #     })

    #     # Increment epoch count and save to file
    #     current_epoch += 1
    #     with open(epoch_file, "w") as f:
    #         f.write(str(current_epoch))

    #     return results

    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)

        student_loss = self.student_loss_fn(y, y_pred)

        # Update and gather metrics correctly (Corrected)
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}  # Use self.metrics directly
        # Flatten metric results if they are dictionaries:
        for key, value in results.items():
            if isinstance(value, dict): # Check if the value is a dictionary
                try:
                    results[key] = value['value'] # if it is a dictionary, extract the value
                except KeyError:
                    results[key] = tf.reduce_mean(list(value.values())) # if it is a dictionary and does not have the value key, take the mean of the values

        results.update({"student_loss": student_loss})
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
# Multi-Teacher Knowledge Distillation Execution
##############################################################################
def perform_multi_teacher_distillation():
    """Perform multi-teacher knowledge distillation when both models are trained."""

    # Load dataset
    x_train, y_train, x_test, y_test = load_mnist_dataset()

    # Load teacher models
    teacher_model_A = keras.models.load_model("Model_A-round-10-weights.h5")
    teacher_model_B = keras.models.load_model("Model_B-round-10-weights.h5") 
    teacher_models = [teacher_model_A, teacher_model_B]
    feature_importance_A = calculate_feature_importance(teacher_model_A, x_train, y_test)
    feature_importance_B = calculate_feature_importance(teacher_model_B, x_train, y_test)
   
    # Ensure both arrays have the same shape
    assert feature_importance_A.shape == feature_importance_B.shape, "Feature importance arrays must have the same shape"

    # Convert to TensorFlow tensors
    feature_importance_A_tf = tf.convert_to_tensor(feature_importance_A, dtype=tf.float32)
    feature_importance_B_tf = tf.convert_to_tensor(feature_importance_B, dtype=tf.float32)

    # Compute the average feature importance using tf.reduce_mean
    average_feature_importance = tf.reduce_mean(tf.stack([feature_importance_A_tf, feature_importance_B_tf]), axis=0) # Stack and average the tensors

    # Convert back to NumPy if needed
    average_feature_importance_np = average_feature_importance.numpy()
    
    for teacher in teacher_models:
        teacher.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Initialize Multi-Teacher Distiller
    distiller = MultiTeacherDistiller(student=create_student_model(), teachers=teacher_models, feature_importance=average_feature_importance_np,epsilon=0.01,current_epoch=1)
    
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )
    
    print("\n[Server] Training student model using Multi-Teacher Distillation...\n")
    batch_size = 32
    steps_per_epoch = int(len(x_train) / batch_size)
    distiller.fit(x_train, y_train, epochs=5, batch_size=int(32), steps_per_epoch=steps_per_epoch)

    # Save the final distilled student model
    distiller.student.save("final_student_model.h5")
    print("\n[Server] Multi-Teacher Distillation completed. Student model saved as 'final_student_model.h5'.")

    # Evaluate the model
    final_student_model = keras.models.load_model("final_student_model.h5", compile=False)
    final_student_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("\n[Server] Evaluating final student model on test dataset...")
    test_loss, test_accuracy = final_student_model.evaluate(x_test, y_test, verbose=1)
    print(f"\n[Server] Final Student Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

##############################################################################
# Multi-teacher distillation weights for the final student model
##############################################################################
# Define Model A
def create_model_A():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    return model

# # Define Model A
# def create_model_A():
#     model = keras.Sequential([
#         keras.Input(shape=(28, 28, 1)),  # Input layer

#         # Convolutional layers with Batch Normalization
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
#         layers.BatchNormalization(),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(pool_size=(2, 2)),

#         layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
#         layers.BatchNormalization(),
#         layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(pool_size=(2, 2)),

#         layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
#         layers.BatchNormalization(),
#         layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D(pool_size=(2, 2)),

#         layers.Flatten(),
        
#         # Fully connected layers with Dropout
#         layers.Dense(512, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(256, activation="relu"),
#         layers.Dropout(0.5),
        
#         # Output layer
#         layers.Dense(10, activation="softmax"),
#     ])
#     return model

# Define Model B
def create_model_B():
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

    process_A = multiprocessing.Process(target=start_server, args=(8080, "Model_A"))
    process_B = multiprocessing.Process(target=start_server, args=(8081, "Model_B"))

    process_A.start()
    process_B.start()