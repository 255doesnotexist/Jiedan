import numpy as np
from collections import defaultdict, deque
from tqdm import tqdm

class QLearning:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.experience_replay = deque(maxlen=1000)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.q_table[tuple(state)])]

    def update(self, state, action, reward, next_state):
        self.experience_replay.append((state, action, reward, next_state))
        if len(self.experience_replay) >= 32:
            self.batch_update()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def batch_update(self):
        mini_batch = np.random.choice(len(self.experience_replay), 32, replace=False)
        for idx in mini_batch:
            state, action, reward, next_state = self.experience_replay[idx]
            current_q = self.q_table[tuple(state)][self.actions.index(action)]
            next_max_q = np.max(self.q_table[tuple(next_state)])
            new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
            self.q_table[tuple(state)][self.actions.index(action)] = new_q

class Gridworld:
    def __init__(self):
        self.height = 5
        self.width = 5
        self.state = [0, 0]
        self.actions = ['up', 'down', 'left', 'right']
        self.special_states = {
            'A': {'position': [1, 1], 'reward': 20},
            'B': {'position': [3, 3], 'reward': 10}
        }
        self.max_steps = 50
        self.visited_special = set()

    def reset(self):
        self.state = [np.random.randint(0, self.height), np.random.randint(0, self.width)]
        self.steps = 0
        self.visited_special = set()
        return self.state

    def step(self, action):
        self.steps += 1
        
        next_state = self.state.copy()
        if action == 'up':
            next_state[0] = max(0, next_state[0] - 1)
        elif action == 'down':
            next_state[0] = min(self.height - 1, next_state[0] + 1)
        elif action == 'left':
            next_state[1] = max(0, next_state[1] - 1)
        elif action == 'right':
            next_state[1] = min(self.width - 1, next_state[1] + 1)

        reward = -0.1

        for special_name, special in self.special_states.items():
            if tuple(next_state) == tuple(special['position']) and special_name not in self.visited_special:
                reward = special['reward']
                self.visited_special.add(special_name)
                break

        self.state = next_state
        done = self.steps >= self.max_steps or len(self.visited_special) == len(self.special_states)
        return self.state, reward, done

def train_agent(env, agent, episodes=50000):
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        if (episode + 1) % 1000 == 0:
            print(f"\nEpisode {episode + 1}, Total Reward: {total_reward:.2f}")

def test_agent(env, agent, episodes=100):
    total_rewards = 0
    for episode in tqdm(range(episodes), desc="Testing"):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward

        total_rewards += episode_reward

    average_reward = total_rewards / episodes
    print(f"\nAverage Reward over {episodes} episodes: {average_reward:.2f}")

# 使用示例
env = Gridworld()
agent = QLearning(env.actions)

# 训练智能体
print("Training agent...")
train_agent(env, agent)

# 测试智能体
print("\nTesting agent...")
test_agent(env, agent)