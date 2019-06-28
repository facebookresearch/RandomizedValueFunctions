# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import random
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from qlearn.commun.utils import initialize_weights
from qlearn.commun.local_mnf_layer import MNFLinear


class AtariMNFDQN(nn.Module):
    def __init__(self, args, input_dim, num_actions):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(input_dim, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = MNFLinear(3136, 512, args.hidden_dim, args.n_hidden, args.n_flows_q, args.n_flows_r, use_cuda=args.cuda)
        self.fc2 = MNFLinear(512, num_actions, args.hidden_dim, args.n_hidden, args.n_flows_q, args.n_flows_r, use_cuda=args.cuda)
        initialize_weights(self)

    def forward(self, x, same_noise=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x, same_noise=same_noise))
        x = self.fc2(x, same_noise=same_noise)
        return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()

    def kldiv(self):
        kldiv1 = self.fc1.kldiv()
        kldiv2 = self.fc2.kldiv()
        return kldiv1 + kldiv2


class AtariMNFAgent(object):
    def __init__(self, args, input_dim, num_actions):
        self.num_actions = num_actions
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.double_q = args.double_q
        self.input_dim = input_dim
        self.kl_coeff = float(args.alpha) / args.replay_buffer_size

        self.online_net = AtariMNFDQN(args, input_dim, num_actions)
        if args.model and os.path.isfile(args.model):
            self.online_net.load_state_dict(torch.load(args.model))
        self.online_net.train()

        self.target_net = AtariMNFDQN(args, input_dim, num_actions)
        self.update_target_net()
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr)
        # self.optimiser = optim.RMSprop(self.online_net.parameters(), lr=args.lr,
        #                                alpha=args.alpha, momentum=args.momentum,
        #                                eps=args.eps_rmsprop)
        if args.cuda:
            self.online_net.cuda()
            self.target_net.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if args.cuda else torch.ByteTensor

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Acts based on single state (no batch)
    def act(self, state, eval=False):
        if eval:
            self.online_net.eval()
        else:
            self.online_net.train()
        state = Variable(self.FloatTensor(state / 255.0))
        return self.online_net(state, same_noise=True).data.max(1)[1][0]

    def learn(self, states, actions, rewards, next_states, terminals):
        self.online_net.train()
        self.online_net.reset_noise()
        self.target_net.eval()
        states = Variable(self.FloatTensor(states / 255.0))
        actions = Variable(self.LongTensor(actions))
        next_states = Variable(self.FloatTensor(next_states / 255.0))
        rewards = Variable(self.FloatTensor(rewards)).view(-1, 1)
        terminals = Variable(self.FloatTensor(terminals)).view(-1, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_values = self.online_net(states, same_noise=False)
        kldiv = self.online_net.kldiv()
        state_action_values = state_values.gather(1, actions.view(-1, 1))

        if self.double_q:
            next_actions = self.online_net(next_states, same_noise=False).max(1)[1]
            next_state_values = self.target_net(next_states).gather(1, next_actions.view(-1, 1))
        else:
            next_state_values = self.target_net(next_states).max(1)[0]

        target_state_action_values = rewards + (1 - terminals) * self.discount * next_state_values.view(-1, 1)

        td_errors = F.smooth_l1_loss(state_action_values, target_state_action_values.detach(), size_average=True)

        loss = td_errors + self.kl_coeff * kldiv

        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(self.online_net.parameters(), 10)
        # for param in self.online_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        self.online_net.reset_noise()
        return td_errors, kldiv, loss
