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
from qlearn.toys.model import MNFDQN


class MNFAgent():
    def __init__(self, args, env):
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.double_q = args.double_q
        self.kl_coeff = 1.0 / args.replay_buffer_size

        self.online_net = MNFDQN(args, self.action_space)
        if args.model and os.path.isfile(args.model):
            self.online_net.load_state_dict(torch.load(args.model))
        self.online_net.train()

        self.target_net = MNFDQN(args, self.action_space)
        self.update_target_net()
        self.target_net.train()
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
    def act(self, state, eval=False):
        if eval:
            self.online_net.eval()
        else:
            self.online_net.train()
            # self.online_net.reset_noise()
        state = Variable(self.FloatTensor(state))
        return self.online_net(state, same_noise=True).data.max(1)[1][0]

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def learn(self, states, actions, rewards, next_states, terminals):
        self.online_net.train()
        self.target_net.eval()
        self.online_net.reset_noise()
        states = Variable(self.FloatTensor(states))
        actions = Variable(self.LongTensor(actions))
        next_states = Variable(self.FloatTensor(next_states))
        rewards = Variable(self.FloatTensor(rewards)).view(-1, 1)
        terminals = Variable(self.FloatTensor(terminals)).view(-1, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken

        state_values = self.online_net(states, same_noise=False)
        kldiv = self.online_net.kldiv()
        state_action_values = state_values.gather(1, actions.view(-1, 1))

        if self.double_q:
            next_actions = self.online_net(next_states, same_noise=False).max(1)[1]
            next_state_values = self.target_net(next_states, same_noise=False).gather(1, next_actions.view(-1, 1))
        else:
            next_state_values = self.target_net(next_states).max(1)[0]

        target_state_action_values = rewards + (1 - terminals) * self.discount * next_state_values.view(-1, 1)

        td_errors = F.smooth_l1_loss(state_action_values, target_state_action_values.detach(), size_average=True)

        loss = td_errors + self.kl_coeff * kldiv

        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()

        return loss
