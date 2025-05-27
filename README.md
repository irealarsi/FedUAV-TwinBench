# FedUAV-TwinBench: A Modular Testbed for Federated Learning and Digital Twin Optimization

FedUAV-TwinBench is a reproducible and extensible Python-based testbed designed to simulate and evaluate Federated Learning (FL), Digital Twin (DT), UAV coordination, semantic communication, and service migration across smart city datasets.

---

## 📌 Features
- **Digital Twin-driven prediction**: Delay, energy, and queue forecasting per client.
- **Federated Deep Reinforcement Learning**: DDPG + FedAvg for multi-client collaboration.
- **Semantic-aware communication**: Fidelity-aware selection and scoring.
- **Energy-efficient service migration**: DT-guided trigger logic.
- **Dataset selector**: Run across `casas`, `env_sensors`, and `visdrone` datasets.
- **Full logging**: Tracks reward, migration, divergence, semantic fidelity, and loss.

---

## 📁 Folder Structure
```
FedUAV-TwinBench/
├── datasets/                # Preprocessed federated datasets
├── modules/                # Core modules: dt, fdr, ddpg, etc.
├── simulations/            # Store logs, screenshots, or results
├── utils/                  # Logger + plotting tools
├── logs/                   # Auto-generated logs (CSV + PNG)
├── main.py                 # Main experiment loop
```

---

## 🚀 How to Run
```bash
# Run with CASA dataset (or visdrone/env_sensors)
python main.py --dataset casas
```

Then generate plots:
```bash
python utils/plotter_semcom.py
python utils/plotter_fl.py
python utils/plotter_reward_energy.py
```

---

## 📊 Sample Results (CASAS Dataset)
| Metric                     | Round 1 | Round 5 |
|---------------------------|---------|---------|
| Avg Reward                | -0.88   | -0.35   |
| Avg Delay (ms)           | 0.19    | 0.08    |
| Energy Consumption (J)   | 0.12    | 0.06    |
| Semantic Fidelity (avg)  | 0.42    | 0.87    |
| Clients Migrated (%)     | 40%     | 10%     |

📂 Output Images:
- `reward_trend.png`
- `semantic_vs_energy.png`
- `fl_divergence_trend.png`

---

## ⚗️ Ablation Study Configuration
To run ablation studies:
- Disable DT predictions → comment out `dt.predict()` and use real-time row values.
- Disable semantic selection → replace `select_clients()` with `random.sample(...)`
- Compare logs and plot differences in:
  - reward_trend.png
  - semantic_fidelity_trend.png
  - fl_divergence_trend.png

---

## 📄 Citation
```
@misc{fedua2025,
  title={FedUAV-TwinBench: A Modular Testbed for Federated Deep Learning and Digital Twin Optimization in UAV-Edge Networks},
  author={Ahmed Arslan et al.},
  year={2025},
  note={https://github.com/irealarsi/FedUAV-TwinBench}
}
```

---

## 📬 Contact
For questions or contributions, feel free to open an issue or email: `ahmad.arslan@ucp.edu.pk`

---
