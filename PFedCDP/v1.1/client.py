# import flwr as fl
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# import numpy as np
# from flwr_datasets import FederatedDataset
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import DirichletPartitioner
# from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

# class ServerToClientDistribution:
#     def __init__(self):
#         self.feature_extractor = None  # W_L,global
#         self.classifier = None         # W_C,server
        
#     def distribute_model(self, cluster_id, round_number):
#         """Distribute cluster-specific model to clients"""
#         return {
#             'feature_extractor': self.feature_extractor,
#             'classifier': self.classifier
#         }

# class FisherInformation:
#     def __init__(self, threshold=0.5):
#         self.threshold = threshold
       
#     def calculate_fisher(self, model, data, labels):
#         """Calculate Fisher Information for each parameter"""
#         fisher_values = {}
#         with tf.GradientTape() as tape:
#             predictions = model(data)
#             loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            
#         gradients = tape.gradient(loss, model.trainable_variables)
#         for param, grad in zip(model.trainable_variables, gradients):
#             fisher_values[param.name] = tf.square(grad)
            
#         return fisher_values
    
#     def create_masks(self, fisher_values):
#         """Create personal and global masks based on Fisher values"""
#         personal_mask = {name: tf.cast(value >= self.threshold, tf.float32) 
#                         for name, value in fisher_values.items()}
#         global_mask = {name: tf.cast(value < self.threshold, tf.float32) 
#                       for name, value in fisher_values.items()}
#         return personal_mask, global_mask

# class PrivacyPreservingFisher:
#     def __init__(self, threshold=0.5, l2_norm_clip=1.0, noise_multiplier=1.1, num_microbatches=1):
#         self.threshold = threshold
#         self.l2_norm_clip = l2_norm_clip
#         self.noise_multiplier = noise_multiplier
#         self.num_microbatches = num_microbatches
#         self.quant_params_history = {}
        
#     def quantize_weights(self, weights, num_bits=8):
#         """Quantize weights to reduced precision"""
#         min_val = np.min(weights)
#         max_val = np.max(weights)
        
#         # Calculate scale and zero point
#         scale = (max_val - min_val) / (2**num_bits - 1)
#         zero_point = round(-min_val / scale)
        
#         # Quantize
#         quantized = np.clip(np.round(weights / scale) + zero_point, 0, 2**num_bits - 1)
        
#         return quantized, {'scale': scale, 'zero_point': zero_point}

#     def calculate_fisher_with_privacy(self, model, data, labels):
#         """Calculate Fisher Information with differential privacy"""
#         # Create DP optimizer wrapper
#         dp_optimizer = dp_optimizer_keras.DPKerasAdamOptimizer(
#             l2_norm_clip=self.l2_norm_clip,
#             noise_multiplier=self.noise_multiplier,
#             num_microbatches=self.num_microbatches,
#             learning_rate=0.001
#         )
        
#         fisher_values = {}
#         quantized_weights = {}
#         self.quant_params_history.clear()

#         with tf.GradientTape() as tape:
#             predictions = model(data)
#             loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            
#         # Get DP gradients
#         gradients = tape.gradient(loss, model.trainable_variables)
#         dp_gradients = dp_optimizer._compute_gradients(
#             loss, model.trainable_variables, tape=tape
#         )

#         # Calculate Fisher values and quantize weights
#         for param, grad in zip(model.trainable_variables, dp_gradients):
#             # Fisher calculation with DP gradients
#             fisher_values[param.name] = tf.square(grad)
            
#             # Quantize weights
#             weight_numpy = param.numpy()
#             quantized, quant_params = self.quantize_weights(weight_numpy)
#             quantized_weights[param.name] = quantized
#             self.quant_params_history[param.name] = quant_params
            
#         return fisher_values, quantized_weights
    
#     def create_masks(self, fisher_values):
#         """Create personal and global masks based on Fisher values"""
#         personal_mask = {name: tf.cast(value >= self.threshold, tf.float32) 
#                         for name, value in fisher_values.items()}
#         global_mask = {name: tf.cast(value < self.threshold, tf.float32) 
#                       for name, value in fisher_values.items()}
#         return personal_mask, global_mask

#     def get_private_weights_for_server(self, model, data, labels):
#         """Get privacy-preserved and quantized weights for server upload"""
#         fisher_values, quantized_weights = self.calculate_fisher_with_privacy(
#             model, data, labels
#         )
#         personal_mask, global_mask = self.create_masks(fisher_values)
        
#         return {
#             'weights': quantized_weights,
#             'quantization_params': self.quant_params_history,
#             'personal_mask': personal_mask,
#             'global_mask': global_mask
#         }

# class KnowledgeDistillation:
#     def __init__(self, temperature=3.0, lambda_balance=0.5):
#         self.temperature = temperature
#         self.lambda_balance = lambda_balance
        
#     def compute_distillation_loss(self, student_logits, teacher_logits, true_labels):
#         """Compute combined distillation and CE loss"""
#         # Softmax with temperature
#         soft_targets = tf.nn.softmax(teacher_logits / self.temperature)
#         soft_student = tf.nn.softmax(student_logits / self.temperature)
        
#         # KL divergence loss
#         kd_loss = tf.keras.losses.KLDivergence()(soft_targets, soft_student)
        
#         # Standard cross-entropy loss
#         ce_loss = tf.keras.losses.categorical_crossentropy(true_labels, 
#                                                          tf.nn.softmax(student_logits))
        
#         # Combined loss
#         total_loss = ((1 - self.lambda_balance) * ce_loss + 
#                      self.lambda_balance * (self.temperature ** 2) * kd_loss)
        
#         return total_loss

# def partition_data(client_id):
#     # Partitioning the data using Dirichlet distribution to ensure non-IID
#     partitioner = DirichletPartitioner(
#         num_partitions=10, partition_by="label",
#         alpha=0.5, min_partition_size=10, self_balancing=True
#     )
#     # Changed dataset to "fashion_mnist" to load FMNIST instead of MNIST
#     fds = FederatedDataset(dataset="fashion_mnist", partitioners={"train": partitioner})

#     partition = fds.load_partition(client_id, split="train")
#     print(partition[client_id])
#     partition_sizes = [
#         len(fds.load_partition(partition_id)) for partition_id in range(10)
#     ]
#     print(sorted(partition_sizes))
#     # Divide data on each node: 80% train, 20% test
#     partition = partition.train_test_split(test_size=0.2)
#     x_train = [np.array(img).reshape(28, 28, 1) for img in partition["train"]["image"]]
#     y_train = np.array(partition["train"]["label"])

#     x_test = [np.array(img).reshape(28, 28, 1) for img in partition["test"]["image"]]
#     y_test = np.array(partition["test"]["label"])

#     # Convert to NumPy arrays and normalize
#     x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0

#     return x_train, y_train, x_test, y_test

# client_id = int(input("Enter client ID for Model A (e.g., 1, 2, 3...): "))


# # Define Model A
# def create_model_A():
#     model = keras.Sequential([
#         keras.Input(shape=(28, 28, 1)),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(10, activation="softmax"),
#     ])
#     return model

# # Load dataset
# x_train, y_train, x_test, y_test = partition_data(client_id)

# # Initialize Model A
# model = create_model_A()
# model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

# # Define Flower client
# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return model.get_weights()

#     def fit(self, parameters, config):
#         model.set_weights(parameters)
#         model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
#         return model.get_weights(), len(x_train), {}

#     def evaluate(self, parameters, config):
#         model.set_weights(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
#         return loss, len(x_test), {"accuracy": accuracy}

# # Connect to server on port 8080
# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
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

# Load dataset
x_train, y_train, x_test, y_test = partition_data(client_id)

# Initialize Model A
model = create_model_A()
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        return loss, len(x_test), {"accuracy": accuracy}

# Connect to server on port 8080
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())