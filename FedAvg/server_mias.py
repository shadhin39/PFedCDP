# server_mias.py
# Federated Learning Server for MIAS Dataset
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
    
    # Weighted precision, recall, and F1 score
    precisions = [num_examples * m.get("precision", 0) for num_examples, m in metrics]
    recalls = [num_examples * m.get("recall", 0) for num_examples, m in metrics]
    f1_scores = [num_examples * m.get("f1_score", 0) for num_examples, m in metrics]
    
    # Weighted AUC, specificity, and sensitivity
    aucs = [num_examples * m.get("auc", 0) for num_examples, m in metrics]
    specificities = [num_examples * m.get("specificity", 0) for num_examples, m in metrics]
    sensitivities = [num_examples * m.get("sensitivity", 0) for num_examples, m in metrics]
    
    weighted_precision = sum(precisions) / total_examples if total_examples > 0 else 0
    weighted_recall = sum(recalls) / total_examples if total_examples > 0 else 0
    weighted_f1 = sum(f1_scores) / total_examples if total_examples > 0 else 0
    weighted_auc = sum(aucs) / total_examples if total_examples > 0 else 0
    weighted_specificity = sum(specificities) / total_examples if total_examples > 0 else 0
    weighted_sensitivity = sum(sensitivities) / total_examples if total_examples > 0 else 0
    
    # Additional metrics if available
    result = {
        "accuracy": weighted_accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1_score": weighted_f1,
        "auc": weighted_auc,
        "specificity": weighted_specificity,
        "sensitivity": weighted_sensitivity
    }
    
    # Log aggregation info
    print(f"\n=== Round Aggregation ===")
    print(f"Total examples: {total_examples}")
    print(f"Weighted accuracy: {weighted_accuracy:.4f}")
    print(f"Weighted precision: {weighted_precision:.4f}")
    print(f"Weighted recall: {weighted_recall:.4f}")
    print(f"Weighted F1 score: {weighted_f1:.4f}")
    print(f"Weighted AUC: {weighted_auc:.4f}")
    print(f"Weighted specificity: {weighted_specificity:.4f}")
    print(f"Weighted sensitivity: {weighted_sensitivity:.4f}")
    print(f"Client accuracies: {[m['accuracy'] for _, m in metrics]}")
    
    return result

# Custom strategy for MIAS
class MIASFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_accuracies = []
        self.round_precisions = []
        self.round_recalls = []
        self.round_f1_scores = []
        
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
        
        # Store round metrics
        if aggregated_metrics:
            if "accuracy" in aggregated_metrics:
                self.round_accuracies.append(aggregated_metrics["accuracy"])
            if "precision" in aggregated_metrics:
                self.round_precisions.append(aggregated_metrics["precision"])
            if "recall" in aggregated_metrics:
                self.round_recalls.append(aggregated_metrics["recall"])
            if "f1_score" in aggregated_metrics:
                self.round_f1_scores.append(aggregated_metrics["f1_score"])
            
        # Print round summary
        print(f"\n=== MIAS Round {server_round} Results ===")
        print(f"Aggregated loss: {aggregated_loss:.4f}")
        print(f"Aggregated accuracy: {aggregated_metrics.get('accuracy', 0):.4f}")
        print(f"Aggregated precision: {aggregated_metrics.get('precision', 0):.4f}")
        print(f"Aggregated recall: {aggregated_metrics.get('recall', 0):.4f}")
        print(f"Aggregated F1 score: {aggregated_metrics.get('f1_score', 0):.4f}")
        print(f"Number of clients: {len(results)}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"mias_round_{server_round}_{timestamp}.txt", "w") as f:
            f.write(f"Round: {server_round}\n")
            f.write(f"Loss: {aggregated_loss}\n")
            f.write(f"Accuracy: {aggregated_metrics.get('accuracy', 0)}\n")
            f.write(f"Precision: {aggregated_metrics.get('precision', 0)}\n")
            f.write(f"Recall: {aggregated_metrics.get('recall', 0)}\n")
            f.write(f"F1 Score: {aggregated_metrics.get('f1_score', 0)}\n")
            f.write(f"Clients: {len(results)}\n")
            
        return aggregated_loss, aggregated_metrics

# Define server configuration
config = ServerConfig(num_rounds=10)  # 10 rounds for medical imaging

# Define strategy with custom aggregation
strategy = MIASFedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=2,  # Minimum 2 clients for training
    min_evaluate_clients=2,  # Minimum 2 clients for evaluation
    min_available_clients=2,  # Wait for at least 2 clients
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
