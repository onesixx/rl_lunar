import os
import time
models_dir = f"p_models/{int(time.time())}/"
logdir = f"p_logs/{int(time.time())}/"
if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# ------------------------------------------------------------------------------
from stable_baselines3 import PPO
from p3_02 import CustomEnv

env = CustomEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")