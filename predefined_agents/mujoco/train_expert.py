from stable_baselines3 import SAC
import gymnasium as gym

env = gym.make('HalfCheetah-v4')
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("predefined_agents/mujoco/half_cheetah_expert")