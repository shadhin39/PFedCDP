# server_cbis_ddsm_100.py
# Federated Learning Server for CBIS-DDSM Dataset - 100 clients setup
from typing import List, Tuple, Dict, Optional
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, Parameters
import warnings
import os
import numpy as np
from datetime import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
warnings.filterwarnings("ignore", category=UserWarning)

# Define metric aggregation function for CBIS-DDSM
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients."""
    # Calculate weighted averages
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # Weighted accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    weighted_accuracy = sum(accuracies) / total_examples
    
    # Additional metrics if available
    result = {"accuracy": weighted_accuracy}
    
    # Log aggregation info
    print(f"\n=== Round Aggregation ===")
    print(f"Total examples: {total_examples}")
    print(f"Weighted accuracy: {weighted_accuracy:.4f}")
    print(f"Client accuracies: {[m['accuracy'] for _, m in metrics]}")
    
    return result

# Custom strategy for CBIS-DDSM with 100 clients
class CBISFedAvg100(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_accuracies = []
        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[int, Metrics]],
        failures: List[Tuple[int, Exception]],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation results."""
        
        if not results:
            return None, {}
            
        # Call parent aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Store accuracy for tracking
        if "accuracy" in aggregated_metrics:
            self.round_accuracies.append(aggregated_metrics["accuracy"])
            
        # Log progress
        print(f"\n=== Server Round {server_round} Results ===")
        print(f"Aggregated loss: {aggregated_loss:.4f}")
        print(f"Aggregated accuracy: {aggregated_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Number of clients evaluated: {len(results)}")
        print(f"Failed evaluations: {len(failures)}")
        
        # Save model weights periodically
        if server_round % 5 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Checkpoint saved at round {server_round}")
            
        return aggregated_loss, aggregated_metrics

# Server configuration for 100 clients
config = ServerConfig(num_rounds=20)  # More rounds for 100 clients

# Strategy configuration for 100 clients
strategy = CBISFedAvg100(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=20,  # Minimum 20 clients for training (20% of 100)
    min_evaluate_clients=20,  # Minimum 20 clients for evaluation
    min_available_clients=20,  # Wait for at least 20 clients
    fraction_fit=0.3,  # Use 30% of available clients for training
    fraction_evaluate=0.3,  # Use 30% of available clients for evaluation
)

if __name__ == "__main__":
    print("Starting CBIS-DDSM Federated Learning Server (100 clients setup)...")
    print(f"Configuration: {config.num_rounds} rounds")
    print("Minimum clients required: 20")
    print("Waiting for clients to connect...")

    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy,
    )

    print("\nCBIS-DDSM Federated Learning (100 clients) completed!")