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
# This function takes a list of tuples (num_examples, Metrics) as input.
# It calculates the weighted average of accuracy for each client based on the num of examples (data points or instances in a dataset) used.
# The result is returned as a Metrics object (a dictionary) with the key "accuracy".
# Multiply accuracy by the number of examples:
# The code computes the accuracy for each client by multiplying the accuracy value (m["accuracy"]) with the number of examples (num_examples).
# Aggregate and return the custom metric (weighted average):


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define config
config = ServerConfig(num_rounds=20)

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