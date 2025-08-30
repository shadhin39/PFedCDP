# server_mias.py
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

# Define metric aggregation function for MIAS
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

# Custom strategy for MIAS
class MIASFedAvg(FedAvg):
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
        
        # Store round accuracy
        if aggregated_metrics and "accuracy" in aggregated_metrics:
            self.round_accuracies.append(aggregated_metrics["accuracy"])
            
        # Print round summary
        print(f"\n=== MIAS Round {server_round} Results ===")
        print(f"Aggregated loss: {aggregated_loss:.4f}")
        print(f"Aggregated accuracy: {aggregated_metrics.get('accuracy', 0):.4f}")
        print(f"Number of clients: {len(results)}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"mias_round_{server_round}_{timestamp}.txt", "w") as f:
            f.write(f"Round: {server_round}\n")
            f.write(f"Loss: {aggregated_loss}\n")
            f.write(f"Accuracy: {aggregated_metrics.get('accuracy', 0)}\n")
            f.write(f"Clients: {len(results)}\n")
            
        return aggregated_loss, aggregated_metrics

# Define server configuration
config = ServerConfig(num_rounds=20)  # 20 rounds for medical imaging

# Define strategy with custom aggregation
strategy = MIASFedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=10,  # Minimum 10 clients for training (20% of 50)
    min_evaluate_clients=10,  # Minimum 10 clients for evaluation
    min_available_clients=10,  # Wait for at least 10 clients
)

# Legacy mode
if __name__ == "__main__":
    print("Starting MIAS Federated Learning Server...")
    print(f"Configuration: {config.num_rounds} rounds")
    print("Waiting for clients to connect...")
    
    start_server(
        server_address="0.0.0.0:8081",  # Different port for MIAS
        config=config,
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy,
    )
    
    print("\nMIAS Federated Learning completed!")
