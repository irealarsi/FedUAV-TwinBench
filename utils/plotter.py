import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

def plot_metrics(log_file="logs/training_log.csv"):
    df = pd.read_csv(log_file)

    # Compute average per round
    grouped = df.groupby("round").agg({
        "reward": "mean",
        "delay": "mean",
        "energy": "mean",
        "migration": "sum"
    }).reset_index()

    # Plot Reward
    plt.figure(figsize=(8, 4))
    sns.lineplot(x="round", y="reward", data=grouped, marker="o")
    plt.title("Average Reward per Round")
    plt.xlabel("Round")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("logs/reward_plot.png")
    plt.close()

    # Plot Delay
    plt.figure(figsize=(8, 4))
    sns.lineplot(x="round", y="delay", data=grouped, marker="o")
    plt.title("Average Delay per Round")
    plt.xlabel("Round")
    plt.ylabel("Delay")
    plt.tight_layout()
    plt.savefig("logs/delay_plot.png")
    plt.close()

    # Plot Energy
    plt.figure(figsize=(8, 4))
    sns.lineplot(x="round", y="energy", data=grouped, marker="o")
    plt.title("Average Energy per Round")
    plt.xlabel("Round")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig("logs/energy_plot.png")
    plt.close()

    # Plot Migrations
    plt.figure(figsize=(8, 4))
    sns.barplot(x="round", y="migration", data=grouped)
    plt.title("Number of Migrations per Round")
    plt.xlabel("Round")
    plt.ylabel("Migrations")
    plt.tight_layout()
    plt.savefig("logs/migration_plot.png")
    plt.close()

    print("\n✅ Plots saved to /logs directory")

if __name__ == "__main__":
    plot_metrics()

