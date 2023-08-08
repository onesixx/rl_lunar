
import inspect
#-------------------------------------------------------------------------------
import gymnasium
env = gymnasium.make('LunarLander-v2',render_mode="human")
env.reset()

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