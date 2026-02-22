# modules/utils.py
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_results(log_dir):
    """
    Generates performance plots from the logged CSV data (Episode-wise).
    """
    episode_csv = os.path.join(log_dir, "episode_metrics.csv")
    if not os.path.exists(episode_csv):
        print(f"No log file found at {episode_csv}")
        return

    df = pd.read_csv(episode_csv)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Reward Curve
    axs[0, 0].plot(df['episode'], df['avg_reward'], color='blue')
    axs[0, 0].set_title('Average Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    
    # 2. Throughput
    axs[0, 1].plot(df['episode'], df['total_throughput'], color='green')
    axs[0, 1].set_title('Total System Throughput (kbps)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('kbps')
    
    # 3. Latency Violations
    axs[1, 0].plot(df['episode'], df['latency_violation_rate'], color='red')
    axs[1, 0].set_title('Latency Violation Rate')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Rate')
    
    # 4. CPU Peak Utilization
    axs[1, 1].plot(df['episode'], df['cpu_peak'], color='orange')
    axs[1, 1].set_title('Peak CPU Utilization')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Utilization')
    
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "training_performance.png")
    plt.savefig(plot_path)
    print(f"Training plots saved to {plot_path}")
    plt.close()

def plot_scalability_results(results_df, log_dir):
    """
    Generates Load vs. Performance plots (w.r.t User Count).
    """
    users = results_df['users']
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. System Throughput for both operators
    axs[0, 0].plot(users, results_df['tp_opA'], marker='o', label='Operator A')
    axs[0, 0].plot(users, results_df['tp_opB'], marker='s', label='Operator B')
    axs[0, 0].set_title('System Throughput per Operator')
    axs[0, 0].set_ylabel('Throughput (kbps)')
    axs[0, 0].legend()
    
    # 2. Service-wise throughput
    axs[0, 1].plot(users, results_df['tp_URLLC'], marker='o', label='URLLC')
    axs[0, 1].plot(users, results_df['tp_eMBB'], marker='s', label='eMBB')
    axs[0, 1].plot(users, results_df['tp_mMTC'], marker='^', label='mMTC')
    axs[0, 1].set_title('Service-wise Throughput')
    axs[0, 1].set_ylabel('Throughput (kbps)')
    axs[0, 1].legend()
    
    # 3. Avg Latency
    axs[0, 2].plot(users, results_df['avg_lat'], marker='o', color='purple')
    axs[0, 2].set_title('Average System Latency')
    axs[0, 2].set_ylabel('Latency (ms)')
    
    # 4. Latency violation rate
    axs[1, 0].plot(users, results_df['vio_rate'], marker='o', color='red')
    axs[1, 0].set_title('Latency Violation Rate')
    axs[1, 0].set_ylabel('Violation Rate')
    
    # 5. CPU utilization
    axs[1, 1].plot(users, results_df['cpu_util'], marker='o', color='orange')
    axs[1, 1].set_title('Average CPU Utilization')
    axs[1, 1].set_ylabel('Utilization')
    axs[1, 1].set_ylim(0, 1.1)

    # 6. Memory Violation (Placeholder or Aggregate)
    axs[1, 2].plot(users, results_df['mem_vio'], marker='o', color='brown')
    axs[1, 2].set_title('Memory Violation Rate')
    axs[1, 2].set_ylabel('Violation Rate')

    for ax in axs.flat:
        ax.set_xlabel('Number of Users')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "scalability_analysis.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Scalability plots saved to {plot_path}")
    plt.close()
