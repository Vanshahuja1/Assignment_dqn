# modules/utils.py
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_results(log_dir):
    """
    Generates performance plots from the logged CSV data.
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
