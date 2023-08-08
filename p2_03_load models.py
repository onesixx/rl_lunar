import os

models_dir = "p_models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
#-------------------------------------------------------------------------------
import gymnasium
from stable_baselines3 import PPO

env = gymnasium.make('LunarLander-v2',render_mode="human")
env.reset()

model_path = f"{models_dir}/250000.zip"
# model = PPO('MlpPolicy', env, verbose=1)
model = PPO.load('MlpPolicy', env, verbose=1)

episodes = 5
for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		# pass observation to model to get predicted action
		action, _states = model.predict(obs)
		# pass action to env and get info back
		obs, rewards, done, info = env.step(action)
		# show the environment on the screen
		env.render()
env.close()