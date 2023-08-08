
import gymnasium
from gymnasium import spaces

env = gymnasium.Env
# env.reset()
class CustomEnv(env):
	def __init__(self, arg1, arg2, ...):
		super(CustomEnv, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
		# Example for using image as input (channel-first; channel-last also works):
		self.observation_space = spaces.Box(
			low=0, high=255,
			shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8
	    )
        # self.action_space = spaces.Discrete(4)
        # observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        # self.total_reward = len(self.snake_position) - 3
	def step(self, action):
		...
		return observation, reward, done, info
	def reset(self):
		...
		return observation  # reward, done, info can't be included


    # --------------------------------------------------------------------------
	def render(self, mode='human'):
		...
	def close (self):
		...

# ------------------------------------------------------------------------------

env.close()