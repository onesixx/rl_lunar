import gymnasium
from gymnasium import spaces
import p3_0u

# p3_u.collision_with_apple([250,250],0)

# p3_u.collision_with_boundaries([0,0])
# p3_u.collision_with_boundaries([500,0])

# p3_u.collision_with_self([250,250])


import numpy as np
import cv2
import random
import time
from collections import deque

env = gymnasium.Env
# env.reset()

SNAKE_LEN_GOAL = 30
class CustomEnv(env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box( low=-500, high=500,
            shape=(5+SNAKE_LEN_GOAL, ), dtype=np.float32)
        # self.action_space = spaces.Discrete(4)
        # observation = [head_x,        head_y,
        #                apple_delta_x, apple_delta_y,
        #                snake_length] +
        #                list(self.prev_actions)
        # self.total_reward = len(self.snake_position) - 3

    def reset(self):
        self.img = np.zeros((500,500,3),dtype='uint8')

        # Initial Snake and Apple position
        self.snake_position = [[250,250],[240,250],[230,250]]
        self.apple_position = [random.randrange(1,50)*10, random.randrange(1,50)*10]
        self.score = 0

        self.prev_button_direction = 1
        self.button_direction      = 1

        self.snake_head = [250,250]

        self.prev_reward = 0

        self.done = False

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1) # to create history

        #### create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation  # reward, done, info can't be included

    def step(self, action):
        self.prev_actions.append(action)
        button_direction = action

        #-----------------------------------------------------------------------
        cv2.imshow('a',self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500,500,3),dtype='uint8')
        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
        t_end = time.time() + 0.05
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = p3_0u.collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
        # On collision kill the snake and print the score
        if p3_0u.collision_with_boundaries(self.snake_head) == 1 or p3_0u.collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500,500,3),dtype='uint8')
            cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',self.img)

            self.done = True
        #-----------------------------------------------------------------------

        self.total_reward = len(self.snake_position) - 3  # default length is 3
        self.reward       = self.total_reward - self.prev_reward
        self.prev_reward  = self.total_reward

        if self.done:
            self.reward = -10
        info = {}

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        #### create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.reward, self.done, info

    # --------------------------------------------------------------------------
    def render(self, mode='human'):
        render =0
    def close (self):
        colse =0
# ------------------------------------------------------------------------------


import inspect
env  = CustomEnv()
dir(env)
