import os

models_dir = "p_models/PPO"
logdir = "p_logs"
#-------------------------------------------------------------------------------
import gymnasium
from stable_baselines3 import PPO

env = gymnasium.make('LunarLander-v2',render_mode="human")
env.reset()

model_path = f"{models_dir}/7800.zip"
model = PPO.load(model_path, env, verbose=1)



episodes = 5
for ep in range(episodes):
    # ep =0
    obs,  _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)            # To get predicted action, pass observation to model
        print(f'action:: {action} '+'\n'+'-'*66)

        obs, rewards, done, info, _ = env.step(action) # pass action to env & get info back
        print(f'obs:: {obs}  \n rewards::{rewards}'+'\n'+'-'*66)

        env.render()  # show the environment on the screen
env.close()
