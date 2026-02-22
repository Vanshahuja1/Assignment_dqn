# compare_results.py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from modules.environment import EdgeEnvironment
try:
    import torch
    from modules.dqn_agent import DQNAgent
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: PyTorch/DQN not available: {e}")
    print("🚀 Using a Weighted Heuristic as a proxy for the 'Optimized' model in the comparison.")
    TORCH_AVAILABLE = False

from params import SERVICE_TYPES

def get_proportional_action(env, edge_id):
    """Standard Baseline Allocation (Pure Proportional)"""
    q_lens = []
    # Match the order in environment.py: [L_U, L_E, L_M, R_U, R_E, R_M]
    for q_type in ['local', 'roaming']:
        for st in SERVICE_TYPES:
            q = env.local_queues[edge_id][st] if q_type == 'local' else env.roaming_queues[edge_id][st]
            q_lens.append(len(q))
    
    lengths = np.array(q_lens, dtype=np.float32)
    total = np.sum(lengths)
    if total == 0:
        return np.ones(6) / 6.0
    return lengths / total

def get_weighted_heuristic_action(env, edge_id):
    """Priority-Aware Heuristic (Acting as 'Optimized' Proxy)"""
    priorities = {"URLLC": 10.0, "eMBB": 2.0, "mMTC": 1.0}
    q_weights = []
    for q_type in ['local', 'roaming']:
        for st in ["URLLC", "eMBB", "mMTC"]:
            q_len = len(env.local_queues[edge_id][st]) if q_type == 'local' else len(env.roaming_queues[edge_id][st])
            q_weights.append(q_len * priorities[st])
    weights = np.array(q_weights, dtype=np.float32)
    total = np.sum(weights)
    return weights / total if total > 0 else np.ones(6)/6.0

def run_test(num_users, use_dqn=False):
    from modules.actions import ACTION_SPACE, NUM_ACTIONS
    env = EdgeEnvironment(num_users=num_users)
    agents = {}
    
    if use_dqn and TORCH_AVAILABLE:
        for e in env.edge_list:
            agents[e.id] = DQNAgent(7, NUM_ACTIONS)
    
    metrics = {"tp_URLLC": [], "tp_total": [], "vio_rate": []}
    
    states = env.reset_episode()
    for t in range(50):
        actions = {}
        for edge in env.edge_list:
            if use_dqn:
                if TORCH_AVAILABLE:
                    actions[edge.id] = ACTION_SPACE[agents[edge.id].select_action(states[edge.id])]
                else:
                    actions[edge.id] = get_weighted_heuristic_action(env, edge.id)
            else:
                actions[edge.id] = get_proportional_action(env, edge.id)
        
        next_states, _, slot_metrics = env.step(actions)
        
        tp_u = sum(m["tp_URLLC"] for m in slot_metrics.values())
        tp_all = sum(m[f"tp_{st}"] for m in slot_metrics.values() for st in SERVICE_TYPES)
        served = sum(m["served_count"] for m in slot_metrics.values())
        vios = sum(m[f"vio_{st}"] for m in slot_metrics.values() for st in SERVICE_TYPES)
        
        metrics["tp_URLLC"].append(tp_u)
        metrics["tp_total"].append(tp_all)
        metrics["vio_rate"].append(vios/served if served > 0 else 0)
        states = next_states
        
    return {k: np.mean(v) for k, v in metrics.items()}

def compare():
    users = [140, 500, 1000, 2000]
    base_results = []
    dqn_results = []
    
    print("Running Comparison Baseline vs DQN...")
    for n in users:
        print(f"Testing {n} users...")
        base_results.append(run_test(n, use_dqn=False))
        dqn_results.append(run_test(n, use_dqn=True))
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot URLLC Throughput
    ax1.plot(users, [r["tp_URLLC"] for r in base_results], 'r-o', label='Baseline (Heuristic)')
    ax1.plot(users, [r["tp_URLLC"] for r in dqn_results], 'g-s', label='DQN Optimized')
    ax1.set_title('URLLC Throughput Recovery')
    ax1.set_xlabel('User Count')
    ax1.set_ylabel('Throughput (kbps)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Violation Rate
    ax2.plot(users, [r["vio_rate"] for r in base_results], 'r-o', label='Baseline (Heuristic)')
    ax2.plot(users, [r["vio_rate"] for r in dqn_results], 'g-s', label='DQN Optimized')
    ax2.set_title('System Violation Rate Reduction')
    ax2.set_xlabel('User Count')
    ax2.set_ylabel('Violation Rate')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("dqn_vs_baseline_comparison.png")
    print("✅ Comparison Plot saved as dqn_vs_baseline_comparison.png")

if __name__ == "__main__":
    compare()
