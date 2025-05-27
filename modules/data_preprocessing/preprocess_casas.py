import os
import pandas as pd

RAW_DIR = "datasets/casas/raw"
PROCESSED_DIR = "datasets/casas/processed"
CLIENTS_DIR = "datasets/casas/federated_clients"

def parse_data_file(file_path):
    try:
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            engine='python',
	    usecols=[0, 1, 2],  # ⬅️ Force only 3 columns
            header=None,
            names=["Timestamp", "SensorID", "SensorValue"],
            parse_dates=["Timestamp"]
        )
        return df
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return pd.DataFrame()

def process_all_homes():
    clients = {}
    for home_name in os.listdir(RAW_DIR):
        home_path = os.path.join(RAW_DIR, home_name)
        data_file = os.path.join(home_path, "data")
        if os.path.isdir(home_path) and os.path.exists(data_file):
            print(f"📦 Processing: {home_name}")
            df = parse_data_file(data_file)
            if not df.empty:
                df["Client"] = home_name
                clients[home_name] = df
    return clients

def save_processed_data(clients):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_data = pd.concat(clients.values(), ignore_index=True)
    all_data.to_csv(os.path.join(PROCESSED_DIR, "combined_data.csv"), index=False)
    print("✅ Combined processed data saved.")

    os.makedirs(CLIENTS_DIR, exist_ok=True)
    for i, (name, df) in enumerate(clients.items()):
        client_path = os.path.join(CLIENTS_DIR, f"client_{i}")
        os.makedirs(client_path, exist_ok=True)
        df.to_csv(os.path.join(client_path, "data.csv"), index=False)
    print(f"✅ {len(clients)} client datasets saved.")

def main():
    print("🔍 Reading CASAS homes...")
    clients = process_all_homes()
    if not clients:
        print("⚠ No valid data found.")
        return
    save_processed_data(clients)
    print("🏁 Preprocessing complete.")

if __name__ == "__main__":
    main()
