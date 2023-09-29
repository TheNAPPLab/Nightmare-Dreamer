import gym
import os
import sys
# import panda_gym from s

sys.path.append("Safe-panda-gym/panda_gym")

import panda_gym

import time
env = gym.make("PandaPickAndPlace-v2", render=True)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render(mode='rgb_array')
    # time.sleep(4)

env.close()