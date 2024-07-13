import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 创建Blackjack环境
env = gym.make('Blackjack-v1')

# 基本策略
def basic_strategy(state):
    player_sum, dealer_card, usable_ace = state
    if usable_ace:
        if player_sum >= 19:
            return 0  # 停牌
        else:
            return 1  # 要牌
    else:
        if player_sum >= 17:
            return 0  # 停牌
        elif player_sum <= 11:
            return 1  # 要牌
        else:
            if dealer_card >= 7:
                return 1  # 要牌
            else:
                return 0  # 停牌

# 随机策略
def random_strategy(state):
    return np.random.choice([0, 1])

# 每次访问型蒙特卡洛
def monte_carlo_every_visit(env, num_episodes, policy):
    returns = defaultdict(list)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(num_episodes):
        episode = []
        state, _ = env.reset()  # 确保正确处理reset返回的元组
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)  # 确保正确处理step返回的元组
            episode.append((state, action, reward))
            state = next_state

        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + G
            returns[(state, action)].append(G)
            Q[state][action] = np.mean(returns[(state, action)])

    return Q

# 首次访问型蒙特卡洛
def monte_carlo_first_visit(env, num_episodes, policy):
    returns = defaultdict(list)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(num_episodes):
        episode = []
        state, _ = env.reset()  # 确保只取状态部分
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)  # 确保只取状态部分
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + G
            if (state, action) not in visited:
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
                visited.add((state, action))

    return Q

# 评估策略
def evaluate_policy(env, Q, num_episodes=10000):
    total_return = 0
    for _ in range(num_episodes):
        state, _ = env.reset()  # 正确提取状态
        done = False
        while not done:
            # 确保状态在Q中，否则跳过此状态
            if state in Q:
                action = np.argmax(Q[state])
            else:
                # 如果状态不在Q中，可以随机选择一个动作，或者根据您的策略选择一个默认动作
                action = env.action_space.sample()
            
            state, reward, done, _, _ = env.step(action)  # 修正这里，确保正确处理所有返回值
            total_return += reward
    return total_return / num_episodes

# 运行实验
num_episodes = 100000

Q_every_visit_random = monte_carlo_every_visit(env, num_episodes, random_strategy)
Q_first_visit_random = monte_carlo_first_visit(env, num_episodes, random_strategy)

Q_every_visit_basic = monte_carlo_every_visit(env, num_episodes, basic_strategy)
Q_first_visit_basic = monte_carlo_first_visit(env, num_episodes, basic_strategy)

# 评估结果
print("每次访问型MC（随机策略）平均回报:", evaluate_policy(env, Q_every_visit_random))
print("首次访问型MC（随机策略）平均回报:", evaluate_policy(env, Q_first_visit_random))
print("每次访问型MC（基本策略）平均回报:", evaluate_policy(env, Q_every_visit_basic))
print("首次访问型MC（基本策略）平均回报:", evaluate_policy(env, Q_first_visit_basic))

# 可视化价值函数
def plot_value_function(Q, title):
    V = defaultdict(float)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    
    X = np.arange(12, 22)
    Y = np.arange(1, 11)
    Z = np.zeros((len(X), len(Y)))
    
    for i, player_sum in enumerate(X):
        for j, dealer_card in enumerate(Y):
            Z[i, j] = V[(player_sum, dealer_card, False)]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z.T, cmap=plt.cm.coolwarm)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    fig.colorbar(surf)
    plt.show()

plot_value_function(Q_every_visit_basic, "每次访问型MC（基本策略）价值函数")
plot_value_function(Q_first_visit_basic, "首次访问型MC（基本策略）价值函数")
