# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
sys.path.append('/home/soopark0221/multiagent/RandomizedValueFunctions/qlearn/toys')

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from qlearn.toys.model import MultiDQN
from swag_misc import SWAG
import swag_utils
from utils import *

class MultiAgent():
    def __init__(self, args, env, idx):
        self.action_space = env.action_space.n
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.double_q = args.double_q

        self.online_net = MultiDQN(args, self.action_space)
        if args.model and os.path.isfile(args.model):
            self.online_net.load_state_dict(torch.load(args.model))
        self.online_net.train()

        self.sample_net = self.online_net
        #hard_update(self.sample_net, self.online_net)
            
        self.target_net = MultiDQN(args, self.action_space)
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

        self.lr = args.lr
        self.swag_net = SWAG(self.online_net)
        self.swag_lr = args.swag_lr
        self.swag_start = args.swag_start
        self.sample_freq = args.sample_freq
        self.swag = (args.alg == 'swag')
        self.max_episode= args.max_episodes

        self.idx = idx
    # Acts based on single state (no batch)
    def act(self, state):
        self.online_net.eval()
        state = Variable(self.FloatTensor(state))
        a = self.online_net(state).data[0][self.idx*2:self.idx*2+2].unsqueeze(0).max(1)[1][0]
        return a

    def swag_act(self, state):
        self.online_net.eval()
        state = Variable(self.FloatTensor(state))
        a = []
        for net in self.swag_net_list:
            a.append(net(state).data[0][self.idx*2:self.idx*2+2].unsqueeze(0))
            #print(f'1 {net(state).data}')
        #print(f'********************')

        a = torch.stack(a, dim=0)
        a = torch.mean(torch.Tensor.float(a), dim=0)
        return a.max(1)[1][0]

    def swag_sample(self):
        self.swag_net_list = []
        for i in range(1):
            self.swag_net.sample(self.sample_net, 0.9)
            self.swag_net_list.append(self.sample_net)

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, state, epsilon=0.01):
        return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def learn(self, states, actions, rewards, next_states, terminals, episode, a_other):
        self.online_net.train()
        self.target_net.eval()

        states = torch.FloatTensor(states)
        #states = Variable(self.FloatTensor(states))
        actions = Variable(self.LongTensor(actions))
        next_states = Variable(self.FloatTensor(next_states))
        rewards = Variable(self.FloatTensor(rewards)).view(-1, 1)
        terminals = Variable(self.FloatTensor(terminals)).view(-1, 1)
        actions = torch.index_select(actions, 1, torch.tensor([self.idx]))
        state_action_values = self.online_net(states).gather(1, self.idx*2 + actions.view(-1, 1))
        if self.double_q:
            out = torch.index_select(self.online_net(next_states), 1, torch.tensor([self.idx*2, self.idx*2+1]))
            next_actions = out.max(1)[1] + self.idx*2
            next_state_values = self.target_net(next_states).gather(1, next_actions.view(-1, 1))
        else:
            next_state_values = self.target_net(next_states).data[0][self.idx*2:self.idx*2+2].unsqueeze(0).max(1)[0]

        # Compute V(s_{t+1}) for all next states.
        target_state_action_values = rewards + (1 - terminals) * self.discount * next_state_values.view(-1, 1)


        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, target_state_action_values.detach())
        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()

        lr = self.schedule(episode)
        self.adjust_learning_rate(self.optimiser, lr)

        if self.swag and (episode+1) > self.swag_start:
            self.swag_net.collect_model(self.online_net)
            #if episode % self.sample_freq == 0:
            #    self.swag_net.sample(self.sample_net, 0.5)

        return loss

    def schedule(self, epoch):
        t = (epoch) / (self.swag_start if self.swag else self.max_episode)
        lr_ratio = self.swag_lr / self.lr if self.swag else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.lr * factor

    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return 