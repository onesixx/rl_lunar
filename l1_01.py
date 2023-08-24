import gymnasium as gym
env = gym.make('LunarLander-v2')
env.reset()
print("sample action: :", env.action_space.sample())
print("action space ::",  env.action_space)
print("obv space ::",       env.observation_space)
print("observation space shape: :", env.observation_space.shape)
print("sample observation: :", env.observation_space.sample())
env.close()



