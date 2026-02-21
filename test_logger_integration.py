# test_logger_integration.py
"""
Example script demonstrating how to integrate the professional MetricsLogger
into the Multi-Operator Edge Simulator training loop.
"""
from modules.logger import MetricsLogger
import params
import random
import time

def simulate_training():
    # 1. Initialize Logger
    logger = MetricsLogger(log_dir=params.LOG_DIR, run_name="baseline_test")
    
    # 2. Save snapshot of parameters for reproducibility
    logger.save_config_snapshot("params.py")
    
    print("Starting mock training loop...")
    
    for ep in range(3):  # Mock 3 episodes
        total_reward = 0
        total_throughput = 0
        violations = 0
        
        for slot in range(10):  # Mock 10 slots
            # --- MOCK SIMULATION STEP ---
            throughput = random.uniform(500, 1500)
            latency = random.uniform(5, 50)
            cpu_util = random.uniform(0.4, 0.9)
            mem_util = random.uniform(0.3, 0.8)
            reward = throughput * 0.1 - (latency > 20) * 10
            
            # 3. Log Slot Metrics
            logger.log_slot(slot, ep, {
                "throughput_kbps": throughput,
                "latency_ms": latency,
                "cpu_util": cpu_util,
                "mem_util": mem_util,
                "reward": reward
            })
            
            total_reward += reward
            total_throughput += throughput
            if latency > 20: violations += 1
            
        # 4. Log Episode Aggregates
        logger.log_episode(ep, {
            "avg_reward": total_reward / 10,
            "total_throughput": total_throughput,
            "latency_violation_rate": violations / 10,
            "cpu_peak": 0.95,  # example
            "loss": random.uniform(0.01, 0.1)
        })
        
        print(f"Episode {ep} complete. Reward: {total_reward:.2f}")

    # 5. Finalize
    logger.finalize()

if __name__ == "__main__":
    simulate_training()
