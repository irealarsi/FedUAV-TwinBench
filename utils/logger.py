import csv
import os

def init_logger(log_dir="logs", filename="training_log.csv"):
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["round", "client", "step", "reward", "delay", "energy", "migration"])
    return filepath

def log_step(filepath, round_id, client_id, step, reward, delay, energy, migration_flag):
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round_id, client_id, step, reward, delay, energy, int(migration_flag)])

def log_semcom(round_id, client_id, step, semantic_score, energy):
    path = os.path.join("logs", "semantic_log.csv")
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["round", "client", "step", "semantic_score", "energy"])
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round_id, client_id, step, semantic_score, energy])


def log_divergence(round_id, client_id, divergence_value):
    path = os.path.join("logs", "fl_divergence.csv")
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["round", "client", "divergence"])
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([round_id, client_id, divergence_value])
# Example usage
if __name__ == "__main__":
    path = init_logger()
    log_step(path, round_id=1, client_id=0, step=10, reward=-0.4, delay=0.12, energy=0.08, migration_flag=True)
    print("✅ Logging complete.")

