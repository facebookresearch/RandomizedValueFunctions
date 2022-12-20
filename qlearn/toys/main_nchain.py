# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys
import sys
sys.path.append('/home/soopark0221/multiagent/RandomizedValueFunctions/baselines')

import argparse
import time
from datetime import datetime
import random
import numpy as np
import math
from collections import Counter
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer

from qlearn.toys.agent import Agent
from qlearn.toys.bootstrapped_agent import BootstrappedAgent
from qlearn.toys.bayes_backprop_agent import BayesBackpropAgent
from qlearn.toys.noisy_agent import NoisyAgent
from qlearn.toys.mnf_agent import MNFAgent
from qlearn.envs.nchain import NChainEnv
# from qlearn.toys.memory import ReplayBuffer
from qlearn.toys.test import test
from qlearn.envs.grid_envs import ZigZag6x10


parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--seed', type=int, default=510, help='Random seed')
parser.add_argument('--cuda', type=int, default=1, help='use cuda')
parser.add_argument('--max-steps', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps')

parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--replay_buffer_size', type=int, default=int(10000), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--learning-freq', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
parser.add_argument("--learning-starts", type=int, default=32, help="number of iterations after which learning starts")
parser.add_argument('--discount', type=float, default=0.999, metavar='GAMMA', help='Discount factor')
parser.add_argument('--target-update-freq', type=int, default=100, metavar='TAU', help='Number of steps after which to update target network')
parser.add_argument('--lr', type=float, default=0.001, metavar='ETA', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='EPSILON', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--input-dim', type=int, default=8, help='the length of chain environment')
parser.add_argument('--evaluation-interval', type=int, default=10, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--nheads', type=int, default=10, help='number of heads in Bootstrapped DQN')
parser.add_argument('--agent', type=str, default='DQN', help='type of agent')
parser.add_argument('--final-exploration', type=float, default=0.1, help='last value of epsilon')
parser.add_argument('--final-exploration-step', type=float, default=1000, help='horizon of epsilon schedule')
parser.add_argument('--max-episodes', type=int, default=int(2e3), metavar='EPISODES', help='Number of training episodes')
parser.add_argument('--hidden_dim', type=int, default=int(16), help='number of hidden unit used in normalizing flows')
parser.add_argument('--n-hidden', type=int, default=int(0), help='number of hidden layer used in normalizing flows')
parser.add_argument('--n-flows-q', type=int, default=int(1), help='number of normalizing flows using for the approximate posterior q')
parser.add_argument('--n-flows-r', type=int, default=int(1), help='number of normalizing flows using for auxiliary posterior r')
parser.add_argument('--logdir', type=str, default='logs', help='log directory')
parser.add_argument('--double-q', type=int, default=1, help='whether or not to use Double DQN')

parser.add_argument('--swag_start', default=1000, type=int, help='')
parser.add_argument('--swag_lr', default=0.0001, type=float, help='')
parser.add_argument('--sample_freq', default=30, type=int, help='')
parser.add_argument('--alg', default='ddpg', type=str, help='agent algorithm [ddpg, swag, wol]')
parser.add_argument('--discrete', action='store_true', help='discrete env')
parser.add_argument('--env', default='nchain', type=str, help='[nchain, lava]')

# Setup
args = parser.parse_args()
assert args.agent in ['DQN', 'BootstrappedDQN', 'NoisyDQN', 'BayesBackpropDQN', 'MNFDQN']

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Environment
if args.env == 'nchain':
    env = NChainEnv(args.input_dim)
elif args.env == 'lava':
    env = ZigZag6x10(max_steps=300, act_fail_prob=0, goal=(5, 9), numpy_state=False)
action_space = env.action_space.n

# Log
date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/{}-{}-{}'.format(args.logdir, args.env, args.agent, date)

log = SummaryWriter(run_dir)
print('Writing logs to {}'.format(run_dir))

# Agent
if args.agent == 'BootstrappedDQN':
    dqn = BootstrappedAgent(args, env)
elif args.agent == 'NoisyDQN':
    dqn = NoisyAgent(args, env)
elif args.agent == 'BayesBackpropDQN':
    dqn = BayesBackpropAgent(args, env)
elif args.agent == 'MNFDQN':
    dqn = MNFAgent(args, env)
else:
    dqn = Agent(args, env)

replay_buffer = ReplayBuffer(args.replay_buffer_size)
# mem = ReplayBuffer(args.memory_capacity)

# schedule of epsilon annealing
exploration = LinearSchedule(args.final_exploration_step, args.final_exploration, 1)

# import pdb
# pdb.set_trace()

# Training loop
dqn.online_net.train()
timestamp = 0
for episode in range(args.max_episodes):

    #epsilon = exploration.value(episode)
    epsilon = 0.01
    state, done = env.reset(), False
    if args.agent == 'BootstrappedDQN':
        k = random.randrange(args.nheads)
    elif args.agent == 'VariationalDQN':
        dqn.online_net.freeze_noise()
    elif args.agent == 'BayesBackpropDQN':
        dqn.online_net.reset_noise()
    elif args.agent == 'MNFDQN':
        dqn.online_net.reset_noise()
    
    if args.alg == 'swag' and episode > args.swag_start+50:
        dqn.swag_sample()

    while not done:
        timestamp += 1

        if args.agent == 'BootstrappedDQN':
            action = dqn.act_single_head(state[None], k)
        elif args.agent in ['NoisyDQN', 'BayesBackpropDQN', 'MNFDQN']:
            action = dqn.act(state[None], eval=False)
        elif args.agent == 'DQN':
            if args.alg == 'swag':
                if episode <= args.swag_start+50:
                    action = dqn.act_e_greedy(state[None], epsilon=epsilon)
                else:
                    action = dqn.swag_act(state[None])
            else:
                action = dqn.act_e_greedy(state[None], epsilon=epsilon)
        next_state, reward, done, _ = env.step(int(action))
        # Store the transition in memory
        replay_buffer.add(state, action, reward, next_state, float(done))

        # Move to the next state
        state = next_state
        #
        if timestamp % args.target_update_freq == 0:
            dqn.update_target_net()

    if timestamp > args.learning_starts:
        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
        loss = dqn.learn(obses_t, actions, rewards, obses_tp1, dones, episode)
        log.add_scalar('loss', loss, timestamp)

    # if episode % 10 == 0:
    #     visited = []
    #     for transition in replay_buffer.memory:
    #         visited.append(transition.state.sum())
    #     print(Counter(visited))

    if episode > 4:
        avg_reward = test(args, env, dqn)  # Test
        print('episode: ' + str(episode) + ', Avg. reward: ' + str(round(avg_reward, 4)))
