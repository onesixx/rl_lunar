
import inspect
import numpy as np
#-------------------------------------------------------------------------------
import gymnasium
from stable_baselines3 import A2C

env = gymnasium.make('LunarLander-v2',render_mode="human")

# print(inspect.getsource(env.__init__))
# print(inspect.getsource(gymnasium.utils.RecordConstructorArgs.__init__))

# print(inspect.getsource(env.reset))
# print(inspect.getsource(env.step))

print(inspect.getsource(env.__init__))
def __init__(
    self,
    env: gym.Env,
    max_episode_steps: int,
):
    """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

    Args:
        env: The environment to apply the wrapper
        max_episode_steps: An optional max episode steps (if ``None``, ``env.spec.max_episode_steps`` is used)
    """
    gym.utils.RecordConstructorArgs.__init__(
        self, max_episode_steps=max_episode_steps
    )
    gym.Wrapper.__init__(self, env)

    self._max_episode_steps = max_episode_steps
    self._elapsed_steps = None

print(inspect.getsource(env.reset))
def reset(self, **kwargs):
    """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.
    Args:
        **kwargs: The kwargs to reset the environment with
    Returns:
        The reset environment
    """
    self._elapsed_steps = 0
    return self.env.reset(**kwargs)


print(inspect.getsource(env.step))
def step(self, action):
    """Steps through the environment
    and if the number of steps elapsed exceeds ``max_episode_steps`` then truncate.
    Args:
        action: The environment step action
    Returns:
        The environment step ``(observation, reward, terminated, truncated, info)`` with `truncated=True`
        if the number of steps elapsed >= max episode steps
    """
    observation, reward, terminated, truncated, info = self.env.step(action)
    self._elapsed_steps += 1
    if self._elapsed_steps >= self._max_episode_steps:
        truncated = True
    return observation, reward, terminated, truncated, info



print('----------------------------train----------------------------')
# ------------------------------------------------------------------------------
# for step in range(200):
#     env.render()
#     some_action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(some_action)
# ------------------------------------------------------------------------------
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100)


print('-----------------------------predict---------------------------')
episodes = 5
for ep in range(episodes):
	# ep=0
	print(ep)
	obs, _= env.reset()
	done = False
	while not done:
		# pass observation to model to get predicted action
		action, _states = model.predict(obs)
		# pass action to env and get info back
		obs, rewards, terminated, truncated , info = env.step(action)
		# show the environment on the screen
		env.render()
env.close()