import numpy as np
from flwr.simulation import start_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import medmnist
from medmnist import INFO, Evaluator

def partition_data(num_partitions=3, concentration=0.5):
    """
    Partition ChestMNIST data using Dirichlet distribution for non-IID federated learning.
    
    Args:
        num_partitions (int): Number of client partitions
        concentration (float): Dirichlet concentration parameter (lower = more non-IID)
    
    Returns:
        FederatedDataset: Partitioned dataset
    """
    # Load ChestMNIST dataset
    info = INFO['chestmnist']
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load training data
    train_dataset = DataClass(split='train', download=True, size=28)  # 28x28 pixels
    
    # Convert to format compatible with flwr_datasets
    train_data = train_dataset.imgs
    train_labels = train_dataset.labels.flatten()
    
    # Reshape and normalize
    if len(train_data.shape) == 3:  # Grayscale
        train_data = train_data.reshape(-1, 28, 28, 1)
    else:  # RGB
        train_data = train_data.reshape(-1, 28, 28, 3)
    
    train_data = train_data.astype(np.float32) / 255.0
    
    # Create partitioner
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=concentration,
        min_partition_size=100,
        self_balancing=True
    )
    
    # Create federated dataset structure
    # Note: This is a simplified approach - you may need to adapt based on your specific flwr_datasets version
    federated_dataset = {
        'data': train_data,
        'labels': train_labels,
        'partitioner': partitioner,
        'num_classes': n_classes
    }
    
    return federated_dataset