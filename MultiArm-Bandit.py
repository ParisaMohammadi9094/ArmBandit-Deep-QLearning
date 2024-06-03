import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

# Hyperparameters
BANDITS = 3
EPISODES = 1000
EPSILON = 0.1
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 1000
TARGET_UPDATE = 10

# Define the network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(BANDITS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, BANDITS)
        )

    def forward(self, x):
        return self.fc(x)

# Experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN agent
class DQNAgent:
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPSILON
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().view(1, 1)
        else:
            return torch.tensor([[random.randrange(BANDITS)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train(agent, bandit_probs, episodes):
    rewards = []
    for i_episode in range(episodes):
        state = torch.zeros(1, BANDITS)
        total_reward = 0
        for t in range(1, 1000):
            action = agent.select_action(state)
            reward = torch.tensor([1.0 if bandit_probs[action.item()] > random.random() else 0.0], dtype=torch.float)
            next_state = state.clone()
            next_state[0, action.item()] += reward.item()  # Update next_state with reward.item()
            total_reward += reward.item()

            agent.memory.push(state, action, next_state, reward)
            state = next_state

            agent.optimize_model()

            if t % TARGET_UPDATE == 0:
                agent.update_target_net()

        rewards.append(total_reward)
    return rewards

def random_agent(bandit_probs, episodes):
    rewards = []
    for i_episode in range(episodes):
        total_reward = 0
        for t in range(1, 1000):
            action = random.randrange(BANDITS)
            reward = bandit_probs[action] > random.random()
            total_reward += reward
        rewards.append(total_reward)
    return rewards

# Bandit probabilities
bandit_probs = np.random.rand(BANDITS)

# Train DQN agent
dqn_agent = DQNAgent()
dqn_rewards = train(dqn_agent, bandit_probs, EPISODES)

# Simulate random agent
random_rewards = random_agent(bandit_probs, EPISODES)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(dqn_rewards, label='DQN Agent')
plt.plot(random_rewards, label='Random Agent')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN vs Random Agent')
plt.legend()
plt.show()

# Moving average for smoother visualization
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 50
plt.figure(figsize=(12, 6))
plt.plot(moving_average(dqn_rewards, window_size), label='DQN Agent (Moving Average)')
plt.plot(moving_average(random_rewards, window_size), label='Random Agent (Moving Average)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN vs Random Agent (Moving Average)')
plt.legend()
plt.show()
