# run_load_test.py
import numpy as np
import pandas as pd
import params
import os
import time
import random
from modules.environment import EdgeEnvironment
from modules.utils import plot_scalability_results

# PDF Constant Imports
from params import SERVICE_TYPES, SERVICE_PI, P_SESSION_REQUEST, AREA_WIDTH, AREA_HEIGHT

try:
    from modules.dqn_agent import DQNAgent
    from modules.actions import ACTION_SPACE, NUM_ACTIONS
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: PyTorch/DQN not available: {e}")
    print("🚀 Using Heuristic Baseline (Proportional Allocation) for metrics.")
    TORCH_AVAILABLE = False

def get_proportional_action(env, edge_id):
    """
    Standard Proportional Allocation (Baseline-2).
    Purely based on queue length, showing why URLLC starvation occurs.
    """
    q_lengths = []
    for q_type in ['local', 'roaming']:
        for st in SERVICE_TYPES:
            q_len = len(env.local_queues[edge_id][st]) if q_type == 'local' else len(env.roaming_queues[edge_id][st])
            q_lengths.append(q_len)
            
    lengths = np.array(q_lengths, dtype=np.float32)
    total_len = np.sum(lengths)
    
    if total_len == 0:
        return np.ones(6) / 6.0
    
    return lengths / total_len

def run_scalability_test():
    user_counts = [140, 500, 1000, 2000]
    results = []
    run_name = f"load_test_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(params.LOG_DIR, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"🚀 Starting Scalability Analysis: {user_counts} users")
    
    for n_users in user_counts:
        print(f"\n--- Testing {n_users} Users ---")
        env = EdgeEnvironment(num_users=n_users)
        agents = {e.id: DQNAgent(7, NUM_ACTIONS) for e in env.edge_list} if TORCH_AVAILABLE else {}
        
        agg = {"tp_opA": [], "tp_opB": [], "tp_URLLC": [], "tp_eMBB": [], "tp_mMTC": [],
               "lat": [], "vios": [], "served": [], "cpu": [], "mem": []}
        
        states = env.reset_episode()
        for t in range(50):
            actions = {}
            for edge in env.edge_list:
                if TORCH_AVAILABLE:
                    actions[edge.id] = ACTION_SPACE[agents[edge.id].select_action(states[edge.id])]
                else:
                    actions[edge.id] = get_proportional_action(env, edge.id)
            
            _, _, slot_metrics = env.step(actions)
            
            # Operator stats
            opA_tp = sum(m[k] for eid, m in slot_metrics.items() if next(e for e in env.edge_list if e.id == eid).operator_id == 0 for k in ["tp_URLLC", "tp_eMBB", "tp_mMTC"])
            opB_tp = sum(m[k] for eid, m in slot_metrics.items() if next(e for e in env.edge_list if e.id == eid).operator_id == 1 for k in ["tp_URLLC", "tp_eMBB", "tp_mMTC"])
            agg["tp_opA"].append(opA_tp); agg["tp_opB"].append(opB_tp)
            
            for st in SERVICE_TYPES:
                agg[f"tp_{st}"].append(sum(m[f"tp_{st}"] for m in slot_metrics.values()))
            
            agg["lat"].append(np.mean([l for m in slot_metrics.values() for st in SERVICE_TYPES for l in m[f"lat_{st}"]]) if any(m["served_count"]>0 for m in slot_metrics.values()) else 0)
            agg["vios"].append(sum(m[f"vio_{st}"] for m in slot_metrics.values() for st in SERVICE_TYPES))
            agg["served"].append(sum(m["served_count"] for m in slot_metrics.values()))
            agg["cpu"].append(np.mean(list(env.prev_cpu_utilization.values())))
            agg["mem"].append(np.mean([m["mem_vio"] for m in slot_metrics.values()]))
        
        results.append({
            "users": n_users, "tp_opA": np.mean(agg["tp_opA"]), "tp_opB": np.mean(agg["tp_opB"]),
            "tp_URLLC": np.mean(agg["tp_URLLC"]), "tp_eMBB": np.mean(agg["tp_eMBB"]), "tp_mMTC": np.mean(agg["tp_mMTC"]),
            "avg_lat": np.mean(agg["lat"]), "vio_rate": np.sum(agg["vios"])/np.sum(agg["served"]) if np.sum(agg["served"])>0 else 0,
            "cpu_util": np.mean(agg["cpu"]), "mem_vio": np.mean(agg["mem"])
        })
        print(f"Done {n_users}: Total TP={results[-1]['tp_opA']+results[-1]['tp_opB']:.1f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(log_dir, "scalability_data.csv"), index=False)
    from modules.utils import plot_scalability_results
    plot_scalability_results(df, log_dir)
    print(f"✅ Saved to {log_dir}")

if __name__ == "__main__":
    run_scalability_test()
