# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from qlearn.commun.utils import initialize_weights
from qlearn.commun.noisy_layer import NoisyLinear
from qlearn.commun.bayes_backprop_layer import BayesBackpropLinear
from qlearn.commun.local_mnf_layer import MNFLinear


class DQN(nn.Module):
    def __init__(self, args, action_space):
        nn.Module.__init__(self)
        self.features = nn.Sequential(
            nn.Linear(args.input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True)
        )
        self.last_layer = nn.Linear(16, action_space)
        initialize_weights(self)

    def forward(self, x):
        x = self.features(x)
        x = self.last_layer(x)
        return x


class BoostrappedDQN(nn.Module):
    def __init__(self, args, action_space):
        nn.Module.__init__(self)
        # self.features = nn.Sequential(
        #     nn.Linear(args.input_dim, 16),
        #     nn.ReLU(inplace=True)
        # )
        self.nheads = args.nheads
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(args.input_dim, 16),
        nn.ReLU(inplace=True),
        nn.Linear(16, 16),
        nn.ReLU(inplace=True),
        nn.Linear(16, action_space)) for _ in range(args.nheads)])

        initialize_weights(self)

    def forward_single_head(self, x, k):
        # x = self.features(x)
        x = self.heads[k](x)
        return x

    def forward(self, x):
        # x = self.features(x)
        out = []
        for head in self.heads:
            out.append(head(x))
        return out


class MNFDQN(nn.Module):
    def __init__(self, args, action_space):
        nn.Module.__init__(self)
        self.fc1 = MNFLinear(args.input_dim, 16, args.hidden_dim, args.n_hidden, args.n_flows_q, args.n_flows_r, use_cuda=args.cuda)
        self.fc2 = MNFLinear(16, 16, args.hidden_dim, args.n_hidden, args.n_flows_q, args.n_flows_r, use_cuda=args.cuda)
        self.fc3 = MNFLinear(16, action_space, args.hidden_dim, args.n_hidden, args.n_flows_q, args.n_flows_r, use_cuda=args.cuda)

    def forward(self, x, same_noise=False):
        x = F.relu(self.fc1(x, same_noise=same_noise))
        x = F.relu(self.fc2(x, same_noise=same_noise))
        x = self.fc3(x, same_noise=same_noise)
        return x

    def kldiv(self):
        kldiv1 = self.fc1.kldiv()
        kldiv2 = self.fc2.kldiv()
        kldiv3 = self.fc3.kldiv()
        return kldiv1 + kldiv2 + kldiv3

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

    # def forward(self, x, kl=True):
    #     if kl:
    #         x, kldiv1 = self.fc1(x, kl=True)
    #         x = F.relu(x)
    #         x, kldiv2 = self.fc2(x, kl=True)
    #         x = F.relu(x)
    #         x, kldiv3 = self.fc3(x, kl=True)
    #         kldiv = kldiv1 + kldiv2 + kldiv3
    #         return x, kldiv
    #     else:
    #         x = F.relu(self.fc1(x, kl=False))
    #         x = F.relu(self.fc2(x, kl=False))
    #         x = self.fc3(x, kl=False)
    #         return x


class NoisyDQN(nn.Module):
    def __init__(self, args, action_space):
        nn.Module.__init__(self)
        self.fc1 = NoisyLinear(args.input_dim, 16)
        self.fc2 = NoisyLinear(16, 16)
        self.fc3 = NoisyLinear(16, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class BayesBackpropDQN(nn.Module):
    def __init__(self, args, action_space):
        nn.Module.__init__(self)
        self.fc1 = BayesBackpropLinear(args.input_dim, 16)
        self.fc2 = BayesBackpropLinear(16, 16)
        self.fc3 = BayesBackpropLinear(16, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

    def get_reg(self):
        reg = self.fc1.kldiv()
        reg += self.fc2.kldiv()
        reg += self.fc3.kldiv()
        return reg
