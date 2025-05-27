import pandas as pd
import numpy as np
import os
import argparse
import random
import shutil
import importlib.util

from modules.digital_twin.state_predictor import DigitalTwinPredictor
from modules.ddpg.ddpg_agent import DDPGAgent
from modules.ddpg.replay_buffer import ReplayBuffer
from modules.fdr.federated_aggregator import fed_avg
from modules.fdr.client_selector import select_clients
from utils.logger import init_logger, log_step, log_semcom, log_divergence
from modules.semantic_communication.semantic_fidelity import compute_semantic_fidelity
from modules.service_migration.migration_decision import should_migrate
from modules.service_migration.mobility_predictor import predict_next_location
from scipy.spatial.distance import cosine

# Parse arguments
parser = argparse.ArgumentParser(description="Run FedUAV-TwinBench with selected dataset.")
parser.add_argument('--dataset', type=str, choices=['env_sensors', 'casas', 'visdrone'], default='env_sensors')
parser.add_argument('--ablation', type=str, choices=['baseline', 'no_dt', 'no_semcom', 'default'], default='default')
args = parser.parse_args()

dataset_name = args.dataset
print(f"\n📂 Running testbed using dataset: {dataset_name}\n")

# Handle ablation config switch
ablation_map = {
    'baseline': 'baseline.py',
    'no_dt': 'no_dt.py',
    'no_semcom': 'no_semcom.py',
    'default': None
}
selected_file = ablation_map[args.ablation]
if selected_file:
    shutil.copyfile(f"simulations/ablation/{selected_file}", "simulations/ablation/config.py")
    print(f"🔁 Using ablation config: {selected_file}")

# Load active ablation config
spec = importlib.util.spec_from_file_location("config", "simulations/ablation/config.py")
ablation_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation_config)
USE_DIGITAL_TWIN = ablation_config.USE_DIGITAL_TWIN
USE_SEMANTIC_SELECTION = ablation_config.USE_SEMANTIC_SELECTION
print(f"🔧 DT Enabled: {USE_DIGITAL_TWIN}, Semantic Selection: {USE_SEMANTIC_SELECTION}\n")

# Parameters
STATE_DIM = 5
ACTION_DIM = 1
ROUNDS = 5
LOCAL_STEPS = 50
BATCH_SIZE = 64
MAX_PARTICIPANTS = 3

# Load federated client data
client_datasets = []
dt_predictors = []

client_path = f"datasets/{dataset_name}/federated_clients"
client_files = sorted([f for f in os.listdir(client_path) if os.path.isdir(os.path.join(client_path, f))])
CLIENTS = len(client_files)

for i in range(CLIENTS):
    path = os.path.join(client_path, f"client_{i}", "data.csv")
    df = pd.read_csv(path)
    print(f"🧾 Columns in {dataset_name} client {i}: {df.columns.tolist()}")

    if dataset_name == "casas":
        df['SensorValue'] = df['SensorValue'].astype(str)
        df['SensorCode'] = df['SensorValue'].astype('category').cat.codes / 100.0

        print(f"📉 After cleaning, client {i} has {len(df)} rows.")
        if len(df) < 3:
            print(f"⚠️ Not enough valid data in client {i}, skipping.")
            continue

        df['rssi'] = df['SensorCode']
        df['cpu_load'] = df['SensorID'].astype('category').cat.codes / 10.0
        df['task_size'] = df['SensorCode'].rolling(window=5, min_periods=1).mean()
        df['queue_length'] = np.abs(df['SensorCode'].diff().fillna(0)) / 10.0
        df['delay'] = 0.05 + df['rssi'] * 0.1 + df['cpu_load'] * 0.1
        df['energy'] = 0.02 + df['task_size'] * 0.2 + df['queue_length'] * 0.1

    elif dataset_name in ["env_sensors", "visdrone"]:
        df = df.rename(columns={'0': 'rssi', '1': 'cpu_load', '2': 'task_size', '3': 'queue_length'})
        df['delay'] = 0.05 + df['rssi'] * 0.1 + df['cpu_load'] * 0.1
        df['energy'] = 0.02 + df['task_size'] * 0.2 + df['queue_length'] * 0.1

    print(f"✅ Client {i} has {len(df)} usable rows after preprocessing.")
    predictor = DigitalTwinPredictor()
    predictor.train(df)

    client_datasets.append(df)
    dt_predictors.append(predictor)

if len(client_datasets) == 0:
    print(f"🚫 No valid clients available. Aborting training.")
    exit(1)

# Global model
global_agent = DDPGAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
log_path = init_logger()

# Federated rounds
for round in range(ROUNDS):
    print(f"\n🌐 Federated Round {round+1}")
    local_weights = []
    semantic_scores = []

    # Select clients
    if USE_SEMANTIC_SELECTION:
        selected_idxs = select_clients(client_datasets, dt_predictors, max_clients=MAX_PARTICIPANTS, strategy="semantic")
    else:
        selected_idxs = random.sample(range(len(client_datasets)), MAX_PARTICIPANTS)

    for i in selected_idxs:
        df = client_datasets[i]
        dt = dt_predictors[i]
        print(f"\n📶 Client {i} Training")

        agent = DDPGAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        agent.actor.load_state_dict(global_agent.actor.state_dict())
        buffer = ReplayBuffer(max_size=10000, state_dim=STATE_DIM, action_dim=ACTION_DIM)

        for step in range(LOCAL_STEPS):
            row = df.sample().iloc[0]

            if USE_DIGITAL_TWIN:
                pred = dt.predict(row['rssi'], row['cpu_load'], row['task_size'], row['queue_length'])
            else:
                pred = {
                    'predicted_delay': row['delay'],
                    'predicted_energy': row['energy'],
                    'predicted_queue': row['queue_length']
                }

            semantic_score = compute_semantic_fidelity(
                features=np.random.rand(1280),
                predicted_delay=pred['predicted_delay'],
                predicted_energy=pred['predicted_energy'],
                object_type="person"
            )
            semantic_scores.append(semantic_score)

            urgency = np.random.uniform(0.2, 1.0)
            state = [pred['predicted_delay'], pred['predicted_energy'], pred['predicted_queue'], semantic_score, urgency]
            action = agent.select_action(np.array(state))

            reward = -(state[0] * 2 + state[1]) * action[0]
            next_state = state
            done = should_migrate(pred['predicted_queue'], pred['predicted_energy'])

            buffer.add(state, action, next_state, reward, done)

            log_step(
                filepath=log_path,
                round_id=round + 1,
                client_id=i,
                step=step,
                reward=reward,
                delay=state[0],
                energy=state[1],
                migration_flag=done
            )

            log_semcom(round + 1, i, step, semantic_score, energy=state[1])

            if buffer.size > BATCH_SIZE:
                agent.train(buffer, BATCH_SIZE, round_id=round + 1, client_id=i, global_step=step)

        local_weights.append(agent.actor.state_dict())

    for i, weights in enumerate(local_weights):
        flat_global = np.concatenate([p.data.cpu().numpy().flatten() for p in global_agent.actor.parameters()])
        flat_local = np.concatenate([weights[k].cpu().numpy().flatten() for k in weights])
        divergence = cosine(flat_global, flat_local)
        log_divergence(round + 1, i, divergence)

    if not local_weights:
        print(f"⚠️ No clients participated in round {round+1}, skipping aggregation.")
        continue
    aggregated_weights = fed_avg(local_weights)
    global_agent.actor.load_state_dict(aggregated_weights)

print("\n✅ Federated Training Complete.")
