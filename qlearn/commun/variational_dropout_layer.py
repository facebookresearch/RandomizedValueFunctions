# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from qlearn.commun.norm_flows import MaskedNVPFlow
from qlearn.commun.utils import _norm


class VariationalDropoutLinear(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, n_hidden, n_flows):
        super(VariationalDropoutLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_flows = n_flows
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.n_flows = n_flows
        self.direction = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer('gzero_epsilon', torch.empty(out_features))
        self.gzero_mu = nn.Parameter(torch.Tensor(out_features))
        self.gzero_logsigma = nn.Parameter(torch.Tensor(out_features))
        self.flow = MaskedNVPFlow(out_features, hidden_dim, n_hidden, n_flows)

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        out_stdv = math.sqrt(4.0 / self.out_features)
        stdv2 = math.sqrt(4.0 / (self.in_features + self.out_features))
        self.direction.data.normal_(0, stdv2)
        self.bias.data.normal_(0, out_stdv)
        self.gzero_mu.data.normal_(0, out_stdv)
        self.gzero_logsigma.data.normal_(math.log(0.1), 1e-3 * out_stdv)

    def reset_noise(self):
        self.gzero_epsilon.copy_(torch.randn(self.out_features))
        self.flow.reset_noise()

    def sample_g(self):
        if self.training:
            gzero_sigma = F.softplus(self.gzero_logsigma)
            gzero = self.gzero_mu + gzero_sigma * self.gzero_epsilon
            g, logdets = self.flow(gzero)
            logq = - torch.log(gzero_sigma).sum()
            logq -= logdets[0]
            logp = - 0.5 * torch.sum(g * g)
            kldiv = logq - logp
            return g, kldiv
        else:
            gzero = self.gzero_mu
            g = self.flow(gzero)
            return g

    def forward(self, input):
        if self.training:
            g, kldiv = self.sample_g()
            # weight = self.direction * (g.view(-1, 1) / _norm(self.direction, dim=0))
            weight = self.direction / _norm(self.direction, dim=0)
            out = g.view(1, -1) * F.linear(input, weight, self.bias)
            # out = F.linear(input, weight, self.bias)
            return out, kldiv
        else:
            g = self.sample_g()
            weight = self.direction * (g.view(-1, 1) / _norm(self.direction, dim=0))
            out = F.linear(input, weight, self.bias)
            return out
