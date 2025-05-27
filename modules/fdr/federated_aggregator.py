
import copy
import numpy as np

def fed_avg(local_weights):
    """
    Perform Federated Averaging.

    Args:
        local_weights (list of dict): List of model.state_dict() from each client.

    Returns:
        dict: Aggregated global model weights.
    """
    global_weights = copy.deepcopy(local_weights[0])
    for key in global_weights.keys():
        for i in range(1, len(local_weights)):
            global_weights[key] += local_weights[i][key]
        global_weights[key] = global_weights[key] / len(local_weights)
    return global_weights


def weighted_fed_avg(local_weights, local_sizes):
    """
    Perform weighted Federated Averaging based on data size.

    Args:
        local_weights (list of dict): List of model.state_dict() from clients.
        local_sizes (list of int): Number of samples per client.

    Returns:
        dict: Aggregated global weights.
    """
    total_size = sum(local_sizes)
    global_weights = copy.deepcopy(local_weights[0])
    for key in global_weights.keys():
        global_weights[key] = global_weights[key] * (local_sizes[0] / total_size)
        for i in range(1, len(local_weights)):
            global_weights[key] += local_weights[i][key] * (local_sizes[i] / total_size)
    return global_weights


# Example usage (requires PyTorch model.state_dicts)
if __name__ == "__main__":
    print("✅ Federated Aggregator ready for model weight integration.")
