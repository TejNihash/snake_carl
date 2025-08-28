import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt  # <-- for plotting

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.990
batch_size = 64
memory_size = 10000
episodes = 500
max_steps = 500

memory = deque(maxlen=memory_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return torch.argmax(q_values).item()

def replay():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
    
    current_q = policy_net(states).gather(1, actions)
    next_q = policy_net(next_states).max(1)[0].unsqueeze(1)
    target_q = rewards + (gamma * next_q * (1 - dones))
    
    loss = loss_fn(current_q, target_q.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# To store total rewards for each episode
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    
    for _ in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        
        replay()
        
        total_reward += reward
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_history.append(total_reward)
    print(f"Episode {episode+1}: Total Reward = {total_reward}, Epsilon = {epsilon:.3f}")

print("Training completed!")

# Plot rewards over episodes
plt.figure(figsize=(12,6))
plt.plot(reward_history, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress on CartPole-v1')
plt.legend()
plt.show()

# Test the trained agent (optional)
state = env.reset()[0]
done = False
while not done:
    action = choose_action(state, 0)  # greedy action, no exploration
    state, reward, done, _, _ = env.step(action)
    env.render()

env.close()
