# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import gym
from gym import spaces
import numpy as np


class NChainEnv(gym.Env):
    """n-Chain environment
    The environment consists of a chain of N states and the agent always starts in state s2,
     from where it can either move left or right.
     In state s1, the agent receives a small reward of r = 0.001 and a larger reward r = 1 in state sN.
     This environment is described in
     Deep Exploration via Bootstrapped DQN(https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn.pdf)
    """
    def __init__(self, n):
        self.n = n
        self.state = 1  # Start at state s2
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.max_nsteps = n + 8

    def step(self, action):
        assert self.action_space.contains(action)
        v = np.arange(self.n)
        reward = lambda s, a: 1.0 if (s == (self.n - 1) and a == 1) else (0.001 if (s == 0 and a == 0) else 0)
        is_done = lambda nsteps: nsteps >= self.max_nsteps

        r = reward(self.state, action)
        if action:    # forward
            if self.state != self.n - 1:
                self.state += 1
        else:   # backward
            if self.state != 0:
                self.state -= 1
        self.nsteps += 1
        s = np.zeros(self.n)
        s[self.state]=1
        #return (v <= self.state).astype('float32'), r, is_done(self.nsteps), None
        return s, r, is_done(self.nsteps), None

    def reset(self):
        v = np.arange(self.n)
        self.state = 1
        self.nsteps = 0
        return (v <= self.state).astype('float32')
