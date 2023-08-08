# https://www.youtube.com/watch?v=XbWhJdQgi7E

import gym
from stable_baselines3 import PPO, A2C
import os

myAlgorithm = "PPO"

model_dir = f"models/{myAlgorithm}"
log_dir =   "logs"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env= gym.make("LunarLander-v2", render_mode="rgb_array")
env.reset()

## Training & Saving
if myAlgorithm =='PPO' :
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
elif myAlgorithm == 'A2C':
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# model.learn(total_timesteps=1000)

TIMESTEPS= 10000
for i in range(1,30):
    print(f"STEPS : {i}")
    model.learn(total_timesteps=TIMESTEPS
                ,reset_num_timesteps=False
                ,tb_log_name=f"{myAlgorithm}")
    model.save(f"{model_dir}/{TIMESTEPS*i}")
# !tensorboard --logdir logs --port 6001


'''episodes =10
for ep in range(episodes):
    print(f'episode {ep}===========================================')
    obs = env.reset()     # instead step
    done = False
    while not done:
        env.render()
        a= env.action_space.sample()
        obs, reward, done, info, etc = env.step(a)
        print(f'action: {a} -- reward : {reward}--{etc}')'''

# 1 episode ------------------------------------------------
# for step in range(200):
#     env.render()
#     a= env.action_space.sample()
#     obs, reward, done, info, etc = env.step(a)
#     print(f'action: {a} -- reward : {reward}--{etc}')


# ---------------------------------------------------------
# print("action space ::", env.action_space)
# print("sample action ::", env.action_space.sample())

# print("obv space ::", env.observation_space)
# print("obv space shape ::", env.observation_space.shape)
# print("obv sample ::", env.observation_space.sample())

env.close()
