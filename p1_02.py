
import inspect
#-------------------------------------------------------------------------------
import gymnasium
env = gymnasium.make('LunarLander-v2',render_mode="human")
# => init reset step
# required before you can step the environment
env.reset()

# (array([ 4.2467116e-04,  1.4092686e+00,  4.3003265e-02, -7.3403455e-02,
#         -4.8535669e-04, -9.7408723e-03,  0.0000000e+00,  0.0000000e+00],
#        dtype=float32),
#  {})

# print(inspect.getsource(env.__init__))
# print(inspect.getsource(gymnasium.utils.RecordConstructorArgs.__init__))

# print(inspect.getsource(env.reset))
# print(inspect.getsource(env.step))

# ------------------------------------------------------------------------------
for step in range(100):
    env.render()
    some_action = env.action_space.sample()
    print(f'action::{some_action}')
    observation, reward, terminated, truncated, info = env.step(some_action)
    print(f'obs::{observation}')
    print(f'reward::{reward}')
    #print(f'obs::{obs}, reward::{reward}, done::{done}, info::{info}')
# ------------------------------------------------------------------------------
env.close()