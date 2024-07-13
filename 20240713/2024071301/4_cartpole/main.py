import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 超参数
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy_net(state).argmax().item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        return states, actions, rewards, next_states, dones

    def update_policy(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train():
    env = gym.make('CartPole-v1')
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    num_episodes = 500

    for episode in range(num_episodes):
        state, _ = env.reset()  # 只获取状态部分
        state = np.array(state)  # 确保状态是numpy数组
        # print(f"Initial state shape: {state.shape}")  # 打印初始状态形状
        total_reward = 0

        for t in range(1, 501):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.array(next_state)  # 确保下一状态是numpy数组
            # print(f"Next state shape: {next_state.shape}")  # 打印下一状态形状
            done = done or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.update_policy()
            state = next_state
            total_reward += reward

            if done:
                break
        
        agent.update_epsilon()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    env.close()

if __name__ == "__main__":
    train()
