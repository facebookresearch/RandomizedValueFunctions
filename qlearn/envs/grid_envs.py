import numpy as np
import sys
from io import StringIO
from typing import Tuple
from gym.envs.toy_text.discrete import DiscreteEnv
import time
from scipy.spatial.distance import cityblock
import torch

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class Specs:
    def __init__(self, max_steps):
        self.max_episode_steps = max_steps


class GridCore(DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape: Tuple[int] = (5, 10), start: Tuple[int] = (0, 0),
                 goal: Tuple[int] = (0, 9), max_steps: int = 100,
                 percentage_reward: bool = False, no_goal_rew: bool = False,
                 dense_reward: bool = False, numpy_state: bool = True,
                 xy_state: bool = False):
        try:
            self.shape = self._shape
        except AttributeError:
            self.shape = shape
        self.nS = np.prod(self.shape, dtype=int)  # type: int
        self.nA = 4
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self._steps = 0
        self._pr = percentage_reward
        self._no_goal_rew = no_goal_rew
        self.total_steps = 0
        self._dr = dense_reward
        self.spec = Specs(max_steps)
        self._nps = numpy_state
        self.xy = xy_state

        P = self._init_transition_probability()

        # We always start in state (3, 0)
        if start is not None:
            isd = np.zeros(self.nS)
            isd[np.ravel_multi_index(start, self.shape)] = 1.0
        else:
            isd = np.ones(self.nS)
            isd[np.ravel_multi_index(goal, self.shape)] = 0.0
            for pit in self._pits:
                isd[pit] = 0.0
            isd *= 1 / (self.nS - len(self._pits) - 1)

        super(GridCore, self).__init__(self.nS, self.nA, P, isd)

    def step(self, a):
        self._steps += 1
        s, r, d, i = super(GridCore, self).step(a)
        if self.xy:
            self.s = np.ravel_multi_index(self.s, self.shape)
        if self._steps >= self.max_steps:
            d = True
            i['early'] = True
            i['needs_reset'] = True
        self.total_steps += 1
        if self._nps:
            #return np.array([s], dtype=np.float32), r, d, i
            return s, r, d, i
        # for one hot state
        s = np.eye(self.observation_space.n)[s]
        return s, r, d, i

    def reset(self):
        self._steps = 0
        self.s = np.random.choice(range(len(self.isd)), p=self.isd)
        self.lastaction = None

        if self.xy:
            s = np.unravel_index(self.s, self.shape)
            if self._nps:
                return np.array([s], dtype=np.float32)
            else:
                return s
        if self._nps:
            return np.array([self.s], dtype=np.float32)
        return self.isd

    def _init_transition_probability(self):
        raise NotImplementedError

    def _check_bounds(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def print_T(self):
        print(self.P[self.s])

    def map_output(self, s, pos):
        if self.s == s:
            output = " x "
        elif pos == self.goal:
            output = " T "
        else:
            output = " o "
        return output

    def map_control_output(self, s, pos):
        return self.map_output(s, pos)

    def map_with_inbetween_goal(self, s, pos, in_between_goal):
        return self.map_output(s, pos)

    def render(self, mode='human', close=False, in_control=None, in_between_goal=None):
        self._render(mode, close, in_control, in_between_goal)

    def _render(self, mode='human', close=False, in_control=None, in_between_goal=None):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        if mode == 'human':
            print('\033[2;0H')

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if in_control:
                output = self.map_control_output(s, position)
            elif in_between_goal:
                output = self.map_with_inbetween_goal(s, position, in_between_goal)
            else:
                output = self.map_output(s, position)
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"
            outfile.write(output)
        outfile.write("\n")
        if mode == 'human':
            if in_control:
                time.sleep(0.2)
            else:
                time.sleep(0.05)


class FallEnv(GridCore):
    _pits = []

    def __init__(self, act_fail_prob: float = 1.0, **kwargs):
        self.afp = act_fail_prob
        super(FallEnv, self).__init__(**kwargs)

    def _calculate_transition_prob(self, current, delta, prob):
        transitions = []
        for d, p in zip(delta, prob):
            new_position = np.array(current) + np.array(d)
            new_position = self._check_bounds(new_position).astype(int)
            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
            if not self._dr:
                reward = -0.01
                is_done = False
                if tuple(new_position) == self.goal:
                    if self._pr:
                        reward = 1 - (self._steps / self.max_steps)
                    elif not self._no_goal_rew:
                        reward = 1.0
                    is_done = True
                elif new_state in self._pits:
                    reward = -1.
                    is_done = True
                transitions.append((p, new_position if self.xy else new_state, reward, is_done))
            else:
                reward = -cityblock(new_position, self.goal)
                is_done = False
                if tuple(new_position) == self.goal:
                    if self._pr:
                        reward = 1 - (self._steps / self.max_steps)
                    elif not self._no_goal_rew:
                        reward = 100.0
                    is_done = True
                elif new_state in self._pits:
                    reward = -100
                    is_done = True
                transitions.append((p, new_position if self.xy else new_state, reward, is_done))
        return transitions

    def _init_transition_probability(self):
        for idx, p in enumerate(self._pits):
            try:
                self._pits[idx] = np.ravel_multi_index(p, self.shape)
            except:
                pass  # <- this has to be here for the agent. Otherwise it throws an unexplainable error
        # Calculate transition probabilities
        P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.nA)}
            other_prob = self.afp / 3.
            tmp = [[UP, DOWN, LEFT, RIGHT],
                   [DOWN, LEFT, RIGHT, UP],
                   [LEFT, RIGHT, UP, DOWN],
                   [RIGHT, UP, DOWN, LEFT]]
            tmp_dirs = [[[-1, 0], [1, 0], [0, -1], [0, 1]],
                        [[1, 0], [0, -1], [0, 1], [-1, 0]],
                        [[0, -1], [0, 1], [-1, 0], [1, 0]],
                        [[0, 1], [-1, 0], [1, 0], [0, -1]]]
            tmp_pros = [[1 - self.afp, other_prob, other_prob, other_prob],
                        [1 - self.afp, other_prob, other_prob, other_prob],
                        [1 - self.afp, other_prob, other_prob, other_prob],
                        [1 - self.afp, other_prob, other_prob, other_prob], ]
            for acts, dirs, probs in zip(tmp, tmp_dirs, tmp_pros):
                P[s][acts[0]] = self._calculate_transition_prob(position, dirs, probs)
        return P

    def map_output(self, s, pos):
        if self.s == s:
            output = " \u001b[33m*\u001b[37m "
        elif pos == self.goal:
            output = " \u001b[37mX\u001b[37m "
        elif s in self._pits:
            output = " \u001b[31m.\u001b[37m "
        else:
            output = " \u001b[30mo\u001b[37m "
        return output

    def map_control_output(self, s, pos):
        if self.s == s:
            return " \u001b[34m*\u001b[37m "
        else:
            return self.map_output(s, pos)

    def map_with_inbetween_goal(self, s, pos, in_between_goal):
        if s == in_between_goal:
            return " \u001b[34mx\u001b[37m "
        else:
            return self.map_output(s, pos)


class ZigZag6x10(FallEnv):
    _pits = [[0, 2], [0, 3],
             [1, 2], [1, 3],
             [2, 2], [2, 3],
             [3, 2], [3, 3],
             [5, 7], [5, 6],
             [4, 7], [4, 6],
             [3, 7], [3, 6],
             [2, 7], [2, 6],
             ]
    _shape = (6, 10)