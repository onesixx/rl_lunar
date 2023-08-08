#import gym
import gymnasium
# Create the environment
#env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env = gymnasium.make('LunarLander-v2')
# required before you can step the environment
env.reset()
# (array([ 4.2467116e-04,  1.4092686e+00,  4.3003265e-02, -7.3403455e-02,
#         -4.8535669e-04, -9.7408723e-03,  0.0000000e+00,  0.0000000e+00],
#        dtype=float32),
#  {})

# ------------------------------------------------------------------------------
print("sample action: :", env.action_space.sample())
# 1
print("action space ::",  env.action_space)
# Discrete(4)

print("obv space ::",       env.observation_space)
# obv space :: Box([-90. -90. -5. -5. -3.1415927 -5. -0. -0.],
#                  [90.   90.  5.  5.  3.1415927  5.  1.  1.], (8,), float32)
print("observation space shape: :", env.observation_space.shape)
# (8,0)

print("sample observation: :", env.observation_space.sample())
# [ 4.9834843  57.936005   -3.9934275   1.8428234  -0.57467085  2.5586786  0.43981507  0.76117915]

env.close()