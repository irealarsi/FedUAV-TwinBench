import os
import pandas as pd
from glob import glob
import numpy as np

RAW_DIR = "datasets/visdrone/raw"
PROCESSED_DIR = "datasets/visdrone/processed"
CLIENTS_DIR = "datasets/visdrone/federated_clients"
N_CLIENTS = 5

def parse_annotations(ann_path):
    files = sorted(glob(os.path.join(ann_path, "*.txt")))
    data = []

    for f in files:
        img_id = os.path.basename(f).replace(".txt", "")
        try:
            with open(f, "r") as file:
                lines = file.readlines()
                obj_count = len(lines)
                data.append({"image_id": img_id, "object_count": obj_count})
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return pd.DataFrame(data)

def split_and_save(df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DIR, "visdrone_summary.csv"), index=False)

    os.makedirs(CLIENTS_DIR, exist_ok=True)
    clients = df.sample(frac=1).reset_index(drop=True)
    splits = np.array_split(clients, N_CLIENTS)

    for i, client_df in enumerate(splits):
        client_path = os.path.join(CLIENTS_DIR, f"client_{i}")
        os.makedirs(client_path, exist_ok=True)
        client_df.to_csv(os.path.join(client_path, "data.csv"), index=False)

def main():
    print("📥 Parsing VisDrone annotations...")
    ann_path = os.path.join(RAW_DIR, "annotations")
    if not os.path.exists(ann_path):
        print("❌ Annotation folder not found.")
        return

    df = parse_annotations(ann_path)
    if df.empty:
        print("⚠ No annotation data parsed.")
        return

    split_and_save(df)
    print("✅ VisDrone preprocessing complete.")

if __name__ == "__main__":
    main()
