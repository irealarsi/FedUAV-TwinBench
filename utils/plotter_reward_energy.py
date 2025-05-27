import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

log_file = "logs/training_log.csv"
if not os.path.exists(log_file):
    raise FileNotFoundError("Training log not found. Please run main.py first.")

df = pd.read_csv(log_file)

# --- Plot 1: Average Reward per Round ---
reward_summary = df.groupby("round")["reward"].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=reward_summary, x="round", y="reward", marker="o")
plt.title("Average Reward per Federated Round")
plt.xlabel("Federated Round")
plt.ylabel("Mean Reward")
plt.tight_layout()
plt.savefig("logs/reward_trend.png")
plt.close()

# --- Plot 2: Energy Consumption per Client ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="client", y="energy", palette="coolwarm")
plt.title("Energy Consumption Distribution per Client")
plt.xlabel("Client ID")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("logs/energy_per_client.png")
plt.close()

# --- Plot 3: Delay Distribution per Client ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="client", y="delay", palette="viridis")
plt.title("Delay Distribution per Client")
plt.xlabel("Client ID")
plt.ylabel("Delay")
plt.tight_layout()
plt.savefig("logs/delay_per_client.png")
plt.close()

# --- Plot 4: Learning Loss per Round (if available) ---
loss_file = "logs/loss_log.csv"
if os.path.exists(loss_file):
    df_loss = pd.read_csv(loss_file)
    loss_summary = df_loss.groupby("round")["loss"].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=loss_summary, x="round", y="loss", marker="o", color="red")
    plt.title("Average Training Loss per Round")
    plt.xlabel("Federated Round")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("logs/loss_trend.png")
    plt.close()
    print("✅ Loss plot saved: logs/loss_trend.png")
else:
    print("ℹ️ Skipped loss plot: logs/loss_log.csv not found.")

print("✅ Reward, Energy, and Delay plots saved to logs/:\n - reward_trend.png\n - energy_per_client.png\n - delay_per_client.png")
