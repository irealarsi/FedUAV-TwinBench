import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

RAW_DIR = "datasets/env_sensors/raw"
PROCESSED_DIR = "datasets/env_sensors/processed"
CLIENTS_DIR = "datasets/env_sensors/federated_clients"
N_CLIENTS = 5  # adjustable


def load_rss_sequences():
    print("📥 Loading RSS sequences...")
    all_files = [f for f in os.listdir(RAW_DIR) if f.startswith("MovementAAL_RSS") and f.endswith(".csv")]
    all_data = []

    for f in all_files:
        path = os.path.join(RAW_DIR, f)
        try:
            df = pd.read_csv(path, comment='#', header=None)
            df["sequence_id"] = int(f.split("_")[-1].replace(".csv", ""))
            all_data.append(df)
        except Exception as e:
            print(f"❌ Failed to read {f}: {e}")

    full_df = pd.concat(all_data, ignore_index=True)
    return full_df

def load_labels():
    label_path = os.path.join(RAW_DIR, "MovementAAL_target.csv")
    if not os.path.exists(label_path):
        raise FileNotFoundError("Missing MovementAAL_target.csv in raw folder.")

    # Skip comment header row and rename columns
    df = pd.read_csv(label_path, comment='#', header=None)
    df.columns = ["sequence_id", "label"]
    return df

def normalize_data(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def split_into_clients(df, n_clients):
    os.makedirs(CLIENTS_DIR, exist_ok=True)
    splits = np.array_split(df.sample(frac=1), n_clients)
    for i, client_df in enumerate(splits):
        client_dir = os.path.join(CLIENTS_DIR, f"client_{i}")
        os.makedirs(client_dir, exist_ok=True)
        client_df.to_csv(os.path.join(client_dir, "data.csv"), index=False)

def main():
    print("🔄 Merging RSS data...")
    rss_df = load_rss_sequences()  # shape: (samples × time × RSS channels)
    labels_df = load_labels()      # shape: (sequence_id, label)

    # Drop rows with any NaNs
    rss_df = rss_df.dropna()

    # Make sequence_id types match
    rss_df["sequence_id"] = rss_df["sequence_id"].astype(int)
    labels_df["sequence_id"] = labels_df["sequence_id"].astype(int)

    # Merge on sequence ID
    merged = rss_df.merge(labels_df, on="sequence_id", how="inner")

    # Normalize RSS values (assumes RSSI features are in columns 0-3)
    feature_cols = [0, 1, 2, 3]  # 4 anchors
    merged = normalize_data(merged, feature_cols)

    # Save full processed version
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    merged.to_csv(os.path.join(PROCESSED_DIR, "env_normalized.csv"), index=False)
    print("✅ Normalized data saved.")

    # Split into federated clients
    split_into_clients(merged, N_CLIENTS)
    print(f"✅ Data split into {N_CLIENTS} federated clients.")

if __name__ == "__main__":
    main()
