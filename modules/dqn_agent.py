# modules/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from params import *

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.fc(state)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LR)
        self.memory = ReplayBuffer(DQN_REPLAY_SIZE)
        
        self.steps_done = 0
        self.epsilon = DQN_EPS_START

    def select_action(self, state):
        self.steps_done += 1
        # Decaying epsilon
        self.epsilon = max(DQN_EPS_END, DQN_EPS_START - self.steps_done / DQN_EPS_DECAY_STEPS)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def update(self):
        if len(self.memory) < DQN_MIN_REPLAY_TO_LEARN:
            return 0.0
            
        states, actions, rewards, next_states = self.memory.sample(DQN_BATCH_SIZE)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Next Q values from target network (Eq. 68)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (DQN_GAMMA * next_q_values)
            
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), DQN_GRAD_CLIP_NORM)
        self.optimizer.step()
        
        # Target network update (Eq. 159 in params.py)
        if self.steps_done % DQN_TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
