# Multi-Operator Edge UPF Scheduling Simulator

A Deep Reinforcement Learning (DQN) based resource optimization system for multi-operator edge computing. This project implements dynamic CPU-splitting and session scheduling across six independent queues per edge server to optimize throughput and minimize latency violations.

---

## 🚩 Milestone Status

| Milestone | Description | Status |
| :--- | :--- | :--- |
| **Milestone 1** | **Core DQN Implementation & Environment Engine** | ✅ **Completed** |


---

## 🛠 What Has Been Done (Milestone 1)

In this milestone, we have transitioned the project from a static topology to a functional Reinforcement Learning environment.

### 1. Mathematical Alignment (PDF Specs)
- **Six-Queue Dynamics**: Implemented three local (backlogged) and three roaming (current-slot) FIFO queues per edge UPF.
- **Serve-or-Forward Rule**: Integrated logic to backlog unserved local sessions and immediately forward unserved roaming sessions to Home UPFs.
- **252-Action Space**: Implemented the quantized simplex constraint ($\Delta = 0.2$), allowing the agent to choose from 252 valid CPU-split vectors.

### 2. DQN Agent Implementation
- Multi-agent per-edge training using PyTorch.
- **State Representation**: 7-dimensional normalized vector (6 queues + 1 previous CPU utilization).
- **Reward Function**: Multi-objective function balancing throughput, latency penalties, CPU waste, and memory violations.

### 3. Professional Logging & Reproducibility
- **Metrics Layer**: Automated CSV/JSON logging of per-slot and per-episode performance.
- **Automated Plotting**: Visual generation of Reward, Throughput, and Latency curves.
- **Config Snapshots**: Automatic versioning of parameters for every simulation run.

---

## 📦 Deliverables

| File | Description |
| :--- | :--- |
| `train_dqn.py` | Main entry point for training and simulation. |
| `modules/environment.py` | The "Physics Engine" (Queue dynamics, Latency model, Eq. 19). |
| `modules/dqn_agent.py` | PyTorch DQN implementation with Replay Buffer. |
| `modules/actions.py` | Generator for the 252 discrete CPU-split actions. |
| `modules/logger.py` | High-performance metrics recorder and snapshot utility. |
| `AUDIT_REPORT.md` | Detailed technical audit of the code against project requirements. |

---

## 🚀 How to Run and Test

### 1. Installation
Ensure you have the required dependencies:
```bash
pip install torch numpy matplotlib pandas
```

### 2. Start Training
Run the training script to see the agents learn in the edge environment:
```bash
python train_dqn.py
```

### 3. Review Results
After execution, results are saved in the `logs/` directory with a unique timestamp:
- **Plots**: View `training_performance.png` for a visual summary of the AI's progress.
- **Metrics**: Open `episode_metrics.csv` for raw data on throughput and violations.
- **Snapshots**: Check the `snapshots/` folder to see the exact settings used for that run.

---

## 📈 Performance Monitoring
The current implementation tracks:
- **System Throughput (kbps)**
- **Latency Violation Rate** (URLLC/eMBB/mMTC)
- **CPU & Memory Utilization**
- **Reward Convergence**
