# train_dqn.py
import numpy as np
import params
from modules.environment import EdgeEnvironment
from modules.dqn_agent import DQNAgent
from modules.actions import ACTION_SPACE, NUM_ACTIONS
from modules.logger import MetricsLogger
from modules.utils import plot_training_results

def train():
    # 1. Setup
    print("Initializing Multi-Operator Edge UPF Scheduling Simulator...")
    env = EdgeEnvironment(num_users=500)
    
    # We use one agent per edge server as per Algorithm 2
    agents = {
        edge.id: DQNAgent(state_dim=7, action_dim=NUM_ACTIONS) 
        for edge in env.edge_list
    }
    
    import time
    run_name = f"dqn_training_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = MetricsLogger(log_dir=params.LOG_DIR, run_name=run_name)
    logger.save_config_snapshot("params.py")
    
    total_episodes = 5  # Set to 5 for a quick demo
    slots_per_episode = params.NUM_SLOTS # 100
    
    print(f"Starting training: {total_episodes} episodes, {slots_per_episode} slots each.")
    
    for ep in range(total_episodes):
        states = env.reset_episode()
        ep_reward = 0
        ep_throughput = 0
        ep_latency_vio = 0
        ep_served = 0
        ep_cpu_util = []
        losses = []
        
        for t in range(slots_per_episode):
            # Select actions for all edges
            actions_indices = {}
            alpha_actions = {}
            for edge_id, agent in agents.items():
                idx = agent.select_action(states[edge_id])
                actions_indices[edge_id] = idx
                alpha_actions[edge_id] = ACTION_SPACE[idx]
            
            # Step environment
            next_states, rewards, slot_metrics = env.step(alpha_actions)
            
            # Store transitions and Update agents
            for edge_id, agent in agents.items():
                agent.memory.push(
                    states[edge_id], 
                    actions_indices[edge_id], 
                    rewards[edge_id], 
                    next_states[edge_id]
                )
                
                # Training step (every N slots)
                if t % params.DQN_TRAIN_EVERY == 0:
                    loss = agent.update()
                    if loss > 0: losses.append(loss)
                
                ep_reward += rewards[edge_id]
            
            # Aggregates for logging
            for e_id in slot_metrics:
                m = slot_metrics[e_id]
                ep_throughput += sum(m[f"tp_{st}"] for st in params.SERVICE_TYPES)
                ep_latency_vio += sum(m[f"vio_{st}"] for st in params.SERVICE_TYPES)
                ep_served += m["served_count"]
                ep_cpu_util.append(env.prev_cpu_utilization[e_id])
            
            # Incremental Slot Logging
            logger.log_slot(t, ep, {
                "avg_reward": np.mean(list(rewards.values())),
                "sys_throughput": sum(sum(m[f"tp_{st}"] for st in params.SERVICE_TYPES) for m in slot_metrics.values()),
                "avg_cpu_util": np.mean(list(env.prev_cpu_utilization.values()))
            })
            
            states = next_states

        # Episode logging
        avg_loss = np.mean(losses) if losses else 0
        logger.log_episode(ep, {
            "avg_reward": ep_reward / (len(agents) * slots_per_episode),
            "total_throughput": ep_throughput,
            "latency_violation_rate": ep_latency_vio / ep_served if ep_served > 0 else 0,
            "cpu_peak": np.max(ep_cpu_util),
            "loss": avg_loss,
            "epsilon": agents[0].epsilon
        })
        
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{total_episodes} | Reward: {ep_reward:.1f} | Epsilon: {agents[0].epsilon:.2f}")

    # 4. Finalize & Plot
    logger.finalize()
    plot_training_results(logger.log_dir)
    
    # Save agents
    for edge_id, agent in agents.items():
        agent.save(f"{logger.log_dir}/agent_edge_{edge_id}.pth")

if __name__ == "__main__":
    train()
