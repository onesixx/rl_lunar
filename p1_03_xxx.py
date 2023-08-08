
import inspect
#-------------------------------------------------------------------------------
import gymnasium
from stable_baselines3 import A2C

env = gymnasium.make('LunarLander-v2',render_mode="human")

# ------------------------------------------------------------------------------
# for step in range(200):
#     env.render()
#     some_action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(some_action)
# ------------------------------------------------------------------------------
model = A2C('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)
model.learn(total_timesteps=200)


print('-----------------------------predict---------------------------')
episodes = 5
for ep in range(episodes):
	print(ep)
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