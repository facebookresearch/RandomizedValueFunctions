# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import torch
from torch.autograd import Variable


# Test DQN
def test(args, env, dqn):
    rewards = []

    # Test performance over several episodes
    done = True
    if args.agent == 'MultiDQN':
        for i in range(2):
            dqn[i].online_net.eval()
    else:
        dqn.online_net.eval()
    # dqn.online_net.freeze_noise()
    for _ in range(args.evaluation_episodes):
        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            if args.agent == 'VariationalDQN':
                action = dqn.act(state[None], sample=False)
            elif args.agent in ['NoisyDQN', 'BayesBackpropDQN', 'MNFDQN']:
                action = dqn.act(state[None], eval=True)
            elif args.agent == 'DQN':
                action = dqn.act(state[None])
            elif args.agent == 'BootstrappedDQN':
                action = dqn.act(state[None]) 
            elif args.agent == 'MultiDQN':
                action = []
                for i in range(2):
                    action.append(dqn[i].act(state[None]))
                   # Choose an action greedily
            state, reward, done, _ = env.step(action if args.agent == 'MultiDQN' else int(action)) # Step
            reward_sum += reward
            if done:
                rewards.append(reward_sum)
                break
    env.close()

    # return average reward
    return sum(rewards) / len(rewards)

