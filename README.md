# FedUAV-TwinBench

**A Digital Twin-Driven Federated Deep Reinforcement Learning Testbed for UAV-IoT Coordination in Smart Cities**

---

## 🌐 Overview

**FedUAV-TwinBench** is a modular, research-oriented testbed that simulates, evaluates, and benchmarks Federated Deep Reinforcement Learning (FDRL) and Digital Twin (DT)-based UAV coordination in IoT-enabled smart city environments. It is designed to optimize real-time decision-making for task offloading, semantic communication, UAV deployment, and energy-aware service migration under mobility constraints.

The system integrates:
- 🧠 Deep Reinforcement Learning (DDPG) for local offloading decisions  
- 🔁 Federated Averaging (FedAvg) for decentralized model aggregation  
- 🛰️ UAV mobility and scheduling based on priority and location  
- 🔬 Digital Twin-based forecasting for proactive resource management  
- 🔉 Semantic-aware compression and fidelity control  
- 📡 Real-world data support from **FedAIoT Benchmark**

---

## 📁 Directory Structure

FedUAV-TwinBench/
├── datasets/
│ ├── drone/
│ │ ├── raw/
│ │ ├── processed/
│ │ └── federated_clients/
│ ├── gas_sensor/
│ │ ├── raw/
│ │ └── processed/
│ └── air_quality/
│ ├── raw/
│ └── processed/
│
├── modules/
│ ├── ddpg/ # Local actor-critic model
│ ├── fdr/ # Federated model aggregation (FedAvg)
│ ├── semantic_communication/ # Semantic encoder and fidelity scoring
│ ├── service_migration/ # Migration decisions based on utility
│ ├── digital_twin/ # Predictive modeling
│ ├── uav_scheduling/ # Priority-aware greedy UAV assignment
│ └── data_preprocessing/ # Dataset loaders, cleaners, and splitters
│
├── models/ # Saved models and checkpoints
├── unity_integration/ # Unity-Azure communication layer
├── simulations/ # NS-3 or synthetic result evaluation
├── utils/ # Logger and config scripts
├── main.py # Entry point to launch experiments
├── requirements.txt # Python dependencies
└── README.md # Project overview
---

## 🚀 Key Features

- ✅ **DDPG**-based offloading at IoT devices  
- ✅ **FedAvg**-based aggregation at UAV edge nodes  
- ✅ **Semantic communication** with fidelity control  
- ✅ **Digital Twin-driven prediction** of air quality and UAV energy  
- ✅ **UAV scheduling and service migration** under resource constraints  
- ✅ **Real-world datasets** from **FedAIoT**  
- ✅ Modular and extendable architecture for Smart City applications  

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/irealarsi/FedUAV-TwinBench.git
cd FedUAV-TwinBench


### 2. Create Python Environment

python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

### 3. Preprocess the Datasets

cd modules\data_preprocessing
python preprocess_drone.py
python preprocess_gas_sensor.py
python preprocess_air_quality.py

### 4. Run Main Testbed

python main.py

📊 Datasets Used
Dataset	Description	Use Case
Drone	UAV flight + sensor logs	Task offloading, DRL state
Gas Sensor	Environmental gas data	Semantic fidelity simulation
Air Quality	Urban air metrics (PM2.5, etc.)	DT prediction and UAV planning

Data sourced from FedAIoT Benchmark

📜 License
Code: MIT License (LICENSE)

Datasets: CC BY-NC 4.0 (datasets/LICENSE)

@misc{FedUAV-TwinBench,
  author = {Ahmad Arslan},
  title = {FedUAV-TwinBench: A Digital Twin-Driven Federated Deep RL Testbed for UAV-IoT Networks},
  year = {2025},
  howpublished = {\url{https://github.com/irealarsi/FedUAV-TwinBench}}
}

👨‍🔬 Contributors
Ahmad Arslan – Lead researcher, architecture and implementation

📬 Contact
GitHub: irealarsi

LinkedIn: Ahmad Arslan

Let me know if you want:
- This content as a downloadable `.md` file.
- To move next into `ddpg_agent.py` or `fedavg.py` development.
- Assistance with pushing this to your GitHub.

You're doing a great job setting up a publishable-grade testbed.

