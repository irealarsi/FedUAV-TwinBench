import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Load divergence log
div_file = "logs/fl_divergence.csv"
if not os.path.exists(div_file):
    raise FileNotFoundError("FL divergence log not found. Run main.py first to generate logs.")

df_div = pd.read_csv(div_file)

# --- Plot 1: Model Divergence per Round ---
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_div, x="round", y="divergence", palette="Blues")
plt.title("Model Divergence Across Rounds")
plt.xlabel("Federated Round")
plt.ylabel("Cosine Divergence")
plt.tight_layout()
plt.savefig("logs/fl_divergence_trend.png")
plt.close()

# --- Plot 2: Client Participation Heatmap ---
client_counts = df_div.groupby(["round", "client"]).size().unstack(fill_value=0)
plt.figure(figsize=(10, 6))
sns.heatmap(client_counts, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Client Participation Across Rounds")
plt.xlabel("Client ID")
plt.ylabel("Federated Round")
plt.tight_layout()
plt.savefig("logs/client_participation_heatmap.png")
plt.close()

print("✅ FL plots saved to logs/:\n - fl_divergence_trend.png\n - client_participation_heatmap.png")
