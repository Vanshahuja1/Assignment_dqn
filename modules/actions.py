# modules/actions.py
import numpy as np

def generate_action_space():
    """
    Generates 252 discrete CPU-split actions.
    Eq. 64: each share takes values in {0, 0.2, 0.4, 0.6, 0.8, 1.0}
    Eq. 66: number of valid vectors is binary coefficient (10 choose 5) = 252.
    """
    actions = []
    quanta = 5  # 1.0 / 0.2
    num_queues = 6
    
    def backtrack(current_combination, remaining_quanta):
        if len(current_combination) == num_queues - 1:
            current_combination.append(remaining_quanta)
            actions.append([q * 0.2 for q in current_combination])
            current_combination.pop()
            return
        
        for q in range(remaining_quanta + 1):
            current_combination.append(q)
            backtrack(current_combination, remaining_quanta - q)
            current_combination.pop()

    backtrack([], quanta)
    return np.array(actions)

# Pre-calculate to export
ACTION_SPACE = generate_action_space()
NUM_ACTIONS = len(ACTION_SPACE)

if __name__ == "__main__":
    print(f"Generated action space with {NUM_ACTIONS} actions.")
    print("Example action:", ACTION_SPACE[0])
