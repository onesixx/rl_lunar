import numpy as np
import pandas as pd
import sys
import os.path as osp
import math
import os
import itertools

import plotly.express as px

import pyarrow as pa
import pyarrow.parquet as pq

# import pa
import argparse
from importlib import import_module
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
from torch import nn

from stable_baselines3 import PPO, DQN, DDPG, SAC, TD3, A2C, HER
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.env_util              import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement 
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
from gymnasium import spaces

LOOKBACK = 5

class CustomNet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_features = observation_space.shape[1]
        self.fc1 = nn.Linear(5*LOOKBACK, 256)
        self.ln1 = nn.LayerNorm(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256, 256)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(256, 256) 

    def forward(self, obs: th.Tensor) -> th.Tensor:
        out = self.fc1(obs)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc_out(out)
        return out
        
        

    # def __init__(self):
    #     self.observation_space = spaces.Box(0, 10, shape=(3,5), dtype=np.float32)
    #     self.action_space = spaces.Box(0, 30, shape=(100,), dtype=np.float32)

    # def _get_obs(self):
    #     return self.observation_space        
   
    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     observation = self._get_obs()
    #     return observation, None

    # def step(self, action):
    #     terminated = None
    #     reward = 1 
    #     observation = self._get_obs()
    #     return observation, reward, terminated, False, None
            
class Env002(gym.Env):
    def __init__(self, df) -> None:
        super().__init__()
        self.df = df
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5*LOOKBACK,), dtype=np.float32) # +1 is additional info        
        self.index = 0
        print(f"=> {self.__class__.__name__} loaded.")
        self.hold_amount = 0
        self.available_cash = 10000000
        self.prev_purse_value = 10000000
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = LOOKBACK
        state = self.df[(self.df.index >= self.index - LOOKBACK) & (self.df.index < self.index)][['open', 'high', 'low', 'close', 'volume']].values
        # state = np.expand_dims(np.ravel(state), axis=0)
        state = np.ravel(state)
        self.available_cash = 10000000
        self.hold_amount = 0
        self.prev_purse_value = 10000000
        return state, None

    def step(self, input_action):
#       print(self.index)
        done = False
        reward = 0
        if len(self.df) == self.index + 1:
            return None, 0, True, False, {}
        price_on_action = self.df[self.df.index == self.index]['close'].values[0]
        price_on_next = self.df[self.df.index == self.index + 1]['close'].values[0]
        current_purse_value = 0
        # action = input_action - 10 
        # action /= 10
        action = input_action[0]
        
        if action < 0:
            if self.hold_amount > 0:
               sell_cnt = math.floor(self.hold_amount * abs(action))
               sell_amt = price_on_action * sell_cnt
               self.hold_amount -= sell_cnt
               self.available_cash += sell_amt
        elif action > 0:
               buy_amt = self.available_cash * abs(action)
               buy_cnt = math.floor(buy_amt / price_on_action)
               buy_amt = price_on_action * buy_cnt
               self.hold_amount += buy_cnt
               self.available_cash -= buy_amt

        current_purse_value = self.available_cash + self.hold_amount * price_on_next
        diff = current_purse_value - self.prev_purse_value
        reward = diff / self.prev_purse_value # * 100
        
        
        self.index += 1
        self.prev_purse_value = current_purse_value
        # state = self.df[self.df.index == self.index][['open', 'high', 'low', 'close', 'volume']].values
        
        
        # state = self.df[(self.df.index >= self.index) & (self.df.index < self.index + 5)][['open', 'high', 'low', 'close', 'volume']].values
        # state = self.df[(self.df.index >= self.index - 5) & (self.df.index < self.index)][['open', 'high', 'low', 'close', 'volume']].values
        # state = np.expand_dims(np.ravel(state), axis=0)
        state = self.df[(self.df.index >= self.index - LOOKBACK) & (self.df.index < self.index)][['open', 'high', 'low', 'close', 'volume']].values
        state = np.ravel(state)
        # state = np.expand_dims(np.ravel(state), axis=0)
        # if state.shape[1] < 25:
        #     done = True    
        # if len(state) < LOOKBACK*5:
        #     done = True
        return state, reward, done, False, {}

    def info(self):
        pass
    
    def render(self, mode):
        print(self.df[self.df.index == self.index]['date'], self.prev_purse_value)
        pass

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learn', action="store_true",)
parser.add_argument('-d', '--data')
args = parser.parse_args()

# args = parser.parse_args(args=['-l', '-d', None])
# print(args.learn)
# print(args.data)

policy_kwargs = dict(
    features_extractor_class=CustomNet,
    features_extractor_kwargs=dict(features_dim=256)
)
    
TICKER='000660'
ALG='PPO'
MODEL_FILE =f'{ALG}-{TICKER}-SIMPLE'

# 96.12.26 ~ 23.06.20   6628
df = pd.read_parquet(f'data/{TICKER}.parquet')
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

# df_sorted_descending = df.sort_values(by='date', ascending=True)
# # fig = px.line(df_sorted_descending, x='date', y=df_sorted_descending.index) # y='close')
# # fig.show()
# dd = df_sorted_descending.reset_index(drop=True)
# dd.to_parquet(f'data/{TICKER}.parquet')

# args.learn = True9
if args.learn:
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df[-2000:-200]
    df = df.reset_index()
    df = df.sort_values('date')
else:
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df[-200:]
    df = df.reset_index()
    df = df.sort_values('date')

vec_env = Env002(df)
obs, _ = vec_env.reset(seed=42)
model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo/", learning_rate=0.0001)

if args.learn:
    if os.path.exists(f'{MODEL_FILE}.zip'):
        model = PPO.load(MODEL_FILE)
        model.set_env(vec_env)
    model.learn(total_timesteps=5000)
    model.save(MODEL_FILE)
    del model # remove to demonstrate saving and loading
        
model = PPO.load(MODEL_FILE, env=vec_env)
obs, _ = vec_env.reset()
while True:
    if obs is None:
        break
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, info = vec_env.step(action)
    vec_env.render("human")