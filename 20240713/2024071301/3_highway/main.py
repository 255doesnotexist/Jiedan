import gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

# 创建Highway环境，并增加并行环境数量
n_envs = 8
env = make_vec_env("highway-v0", n_envs=n_envs)

# 使用MLP Policy进行训练
model = PPO("MlpPolicy", env, verbose=1, n_steps=16, batch_size=128, device='cuda')

# 训练模型
print("Starting training on GPU with MlpPolicy...")
model.learn(total_timesteps=48, reset_num_timesteps=True)
print("Training complete.")

# 保存模型
model.save("ppo_highway_cnn")

# 测试模型
obs = env.reset()
for _ in tqdm(range(100), desc="Testing Progress"):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
print("Testing complete.")
