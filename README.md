# Multi-Operator Edge UPF Scheduling Simulator

A Deep Reinforcement Learning (DQN) based resource optimization system for multi-operator edge computing. This project implements dynamic CPU-splitting and session scheduling across six independent queues per edge server to optimize throughput and minimize latency violations.

---

## 🚩 Milestone Status

| Milestone | Description | Status |
| :--- | :--- | :--- |
| **Milestone 1** | **Core DQN Implementation & Environment Engine** | ✅ **Completed** |

| **Additinally** | **Scalability Analysis & Baseline Comparison** | ✅ **Completed** |

---

## 🛠 What Has Been Done

### Milestone 1: Core Engine & AI Foundation
- **Mathematical Alignment (PDF Specs)**: Implemented the exact 6-queue FIFO dynamics and **Equation 19** queue growth models.
- **252-Action Space**: Integrated the quantized simplex CPU-splitting constraints ($\Delta = 0.2$).
- **DQN Architecture**: Developed a multi-agent system where independent DQN controllers (PyTorch) optimize specific edge UPFs.
- **PSA logic**: Fully implemented Proximal Service Association based on **SINR** attachment to BS.

### Additinally: Scalability & Stress Testing
- **Multi-User Load Testing**: Built a simulation engine to evaluate performance across variable user densities: **140, 500, 1000, and 2000 users**.
- **Granular KPI Tracking**: 
    - Operator-specific Throughput (Operator A vs Operator B).
    - Service-wise Throughput (URLLC vs eMBB vs mMTC).
    - Latency Violation Rates (Eq. 51 alignment).
- **Baseline Comparison (Baseline-2)**: Implemented the Heuristic Proportional Allocation logic to serve as a benchmark against the DQN strategy.
- **Congestion Stress Scenarios**: Tuned arrival rates ($P=0.15$) and CPU budgets to simulate high-congestion environments and validate priority learning.

---

## 📦 Deliverables

| File | Description |
| :--- | :--- |
| `train_dqn.py` | Primary training script for the DQN agents. |
| `run_load_test.py` | Load-testing script for scalability analysis across user counts. |
| `compare_results.py` | Side-by-side comparison script for Baseline vs DQN performance. |
| `modules/environment.py` | Core engine handling Queueing, PSA Association, and Latency modeling. |
| `modules/dqn_agent.py` | AI Logic (DQN, Replay Buffer, Epsilon-greedy exploration). |
| `README.md` | Project summary and execution guide. |

---

## 🚀 How to Run and Test

### 1. Installation
```bash
pip install torch numpy matplotlib pandas
```

### 2. Generate Baseline Scaling Plots
To see how the system performs under increasing user load using standard Heuristic logic:
```bash
python run_load_test.py
```
*Output: `logs/load_test_.../scalability_analysis.png`*

### 3. Generate DQN vs Baseline Comparison
To prove the superiority of the DQN in recovering URLLC throughput:
```bash
python compare_results.py
```
*Output: `dqn_vs_baseline_comparison.png`*

### 4. Continuous Training
To further train the AI agents on the latest environment settings:
```bash
python train_dqn.py
```

---

## 📈 Key Metrics Explained
- **Throughput (kbps)**: Total data successfully served per slot per operator.
- **Violation Rate**: Percentage of sessions missing their latency requirement (e.g., >20ms for URLLC).
- **CPU Utilization**: Average percentage of edge CPU capacity used across all queues.
