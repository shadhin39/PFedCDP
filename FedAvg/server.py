#server.py
from typing import List, Tuple
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import warnings
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
warnings.filterwarnings("ignore", category=UserWarning)

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients with precision, recall, and F1 score."""
    # Calculate weighted averages
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # Weighted accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    weighted_accuracy = sum(accuracies) / total_examples
    
    # Weighted precision, recall, and F1 score (if available)
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
    
    # Aggregate and return custom metrics (weighted average)
    result = {"accuracy": weighted_accuracy}
    
    # Add other metrics if they exist
    if any(m.get("precision", 0) > 0 for _, m in metrics):
        result["precision"] = weighted_precision
    if any(m.get("recall", 0) > 0 for _, m in metrics):
        result["recall"] = weighted_recall
    if any(m.get("f1_score", 0) > 0 for _, m in metrics):
        result["f1_score"] = weighted_f1
    if any(m.get("auc", 0) > 0 for _, m in metrics):
        result["auc"] = weighted_auc
    if any(m.get("specificity", 0) > 0 for _, m in metrics):
        result["specificity"] = weighted_specificity
    if any(m.get("sensitivity", 0) > 0 for _, m in metrics):
        result["sensitivity"] = weighted_sensitivity
    
    return result

# Define config
config = ServerConfig(num_rounds=3)

# Define Strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server
    start_server(
        server_address="0.0.0.0:8080", # address 0.0.0.0 will be able to bind the address from the local network
        config=config,
        grpc_max_message_length = 1024*1024*1024, 
        strategy=strategy,
    )
