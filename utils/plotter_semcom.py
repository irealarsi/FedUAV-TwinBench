import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Semantic Communication Log
log_file = "logs/semantic_log.csv"
if not os.path.exists(log_file):
    raise FileNotFoundError("Semantic log file not found. Run main.py first to generate logs.")

# Read the semantic fidelity log
df = pd.read_csv(log_file)

# --- Plot 1: Semantic Fidelity Score vs. Round ---
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="round", y="semantic_score", ci=None, marker="o")
plt.title("Semantic Fidelity Trend Across Rounds")
plt.xlabel("Federated Round")
plt.ylabel("Semantic Fidelity Score")
plt.tight_layout()
plt.savefig("logs/semantic_fidelity_trend.png")
plt.close()

# --- Plot 2: Energy vs. Semantic Fidelity Tradeoff ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="semantic_score", y="energy", hue="client", palette="tab10")
plt.title("Energy vs. Semantic Fidelity")
plt.xlabel("Semantic Fidelity Score")
plt.ylabel("Energy Consumption")
plt.legend(title="Client", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("logs/semantic_vs_energy.png")
plt.close()

print("✅ Semantic Communication plots saved to logs/:\n - semantic_fidelity_trend.png\n - semantic_vs_energy.png")
