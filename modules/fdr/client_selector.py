import numpy as np
import random

def select_clients(clients, dt_predictors, max_clients=3, strategy="semantic"):
    """
    Select a subset of clients based on semantic importance or randomly.

    Args:
        clients (list of pd.DataFrame): List of client dataframes.
        dt_predictors (list of DigitalTwinPredictor): Corresponding DT predictors.
        max_clients (int): Number of clients to select.
        strategy (str): 'semantic' or 'random'.

    Returns:
        list: Indices of selected clients
    """
    if strategy == "random":
        return random.sample(range(len(clients)), min(max_clients, len(clients)))

    # Semantic-based score: low delay + low energy = high score
    scores = []
    for i, (df, dt) in enumerate(zip(clients, dt_predictors)):
        sample = df.sample().iloc[0]
        pred = dt.predict(
            rssi=sample['rssi'],
            cpu_load=sample['cpu_load'],
            task_size=sample['task_size'],
            queue_length=sample['queue_length']
        )
        score = 1 - (pred['predicted_delay'] + pred['predicted_energy']) / 2  # lower is better
        scores.append((i, score))

    scores.sort(key=lambda x: -x[1])
    selected = [idx for idx, _ in scores[:max_clients]]
    return selected


# Example usage
if __name__ == "__main__":
    print("✅ Client selector ready for integration.")
