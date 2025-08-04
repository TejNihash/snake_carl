import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_DQN(nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        super(CNN_DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # [B, 32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # [B, 64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # [B, 64, 7, 7]
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0  # normalize pixel values from 0–255 to 0–1
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float)
        )
    
    def __len__(self):
        return len(self.buffer)

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = CNN_DQN(state_dim, action_dim)
target_net = CNN_DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())  # Copy weights
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer(10000)

batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
target_update_freq = 10

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state)
        return q_values.argmax().item()

num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(200):  # Max steps per episode
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            
            # Compute current Q values
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * max_next_q_values * (1 - dones)
            
            loss = nn.MSELoss()(q_values, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done:
            break

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.3f}")
