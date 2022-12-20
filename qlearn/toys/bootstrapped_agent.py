# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from qlearn.toys.model import BoostrappedDQN
from collections import Counter


class BootstrappedAgent():
    def __init__(self, args, env):
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.nheads = args.nheads
        self.double_q = args.double_q

        self.online_net = BoostrappedDQN(args, self.action_space)
        if args.model and os.path.isfile(args.model):
            self.online_net.load_state_dict(torch.load(args.model))
        self.online_net.train()

        self.target_net = BoostrappedDQN(args, self.action_space)
        self.update_target_net()
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)
        if args.cuda:
            self.online_net.cuda()
            self.target_net.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if args.cuda else torch.ByteTensor

    # Acts based on single state (no batch)
    def act_single_head(self, state, k):
        self.online_net.eval()
        state = Variable(self.FloatTensor(state))
        return self.online_net.forward_single_head(state, k).data.max(1)[1][0]

    def act(self, state):
        self.online_net.eval()
        state = Variable(self.FloatTensor(state))
        outputs = self.online_net.forward(state)
        actions = []
        for k in range(self.online_net.nheads):
            actions.append(int(outputs[k].data.max(1)[1][0]))
        action, _ = Counter(actions).most_common()[0]
        return action

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, state, k, epsilon=0.01):
        return random.randrange(self.action_space) if random.random() < epsilon else self.act_single_head(state, k)
        #return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def learn(self, states, actions, rewards, next_states, terminals, episode):
        self.online_net.train()
        self.target_net.eval()
        states = Variable(self.FloatTensor(states))
        actions = Variable(self.LongTensor(actions))
        next_states = Variable(self.FloatTensor(next_states))
        rewards = Variable(self.FloatTensor(rewards)).view(-1, 1)
        terminals = Variable(self.FloatTensor(terminals)).view(-1, 1)

        # import pdb
        # pdb.set_trace()
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        online_outputs = self.online_net(states)
        target_outputs = self.target_net(next_states)
        loss = 0
        # import pdb
        # pdb.set_trace()
        for k in range(self.nheads):
            state_action_values = online_outputs[k].gather(1, actions.view(-1, 1))

            # Compute V(s_{t+1}) for all next states.
            if self.double_q:
                next_actions = online_outputs[k].max(1)[1]
                next_state_values = target_outputs[k].gather(1, next_actions.view(-1, 1))
            else:
                next_state_values = target_outputs[k].max(1)[0].view(-1, 1)

            target_state_action_values = rewards + (1 - terminals) * self.discount * next_state_values.view(-1, 1)

            # Compute Huber loss
            loss += F.smooth_l1_loss(state_action_values, target_state_action_values.detach())
        # loss /= args.nheads
        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        return loss
