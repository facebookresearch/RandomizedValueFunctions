# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class BayesBackpropLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_prior=1):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_prior = sigma_prior
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = math.sqrt(3.0 / self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_logsigma.data.fill_(-3)
        self.bias_mu.data.uniform_(-mu_range, -mu_range)
        self.bias_logsigma.data.fill_(-3)

    def reset_noise(self):
        self.weight_epsilon.copy_(torch.randn(self.out_features, self.in_features))
        self.bias_epsilon.copy_(torch.randn(self.out_features))

    def forward(self, input):
        if self.training:
            weight_sigma = F.softplus(self.weight_logsigma)
            bias_sigma = F.softplus(self.bias_logsigma)
            return F.linear(input, self.weight_mu + weight_sigma * self.weight_epsilon, self.bias_mu + bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

    def kldiv(self):
        weight_sigma = F.softplus(self.weight_logsigma)
        bias_sigma = F.softplus(self.bias_logsigma)
        kldiv_weight = torch.sum( math.log(self.sigma_prior) - torch.log(weight_sigma) + \
        (weight_sigma **2 + self.weight_mu **2) /(2 * self.sigma_prior ** 2) - 0.5)

        kldiv_bias = torch.sum( math.log(self.sigma_prior) - torch.log(bias_sigma) + \
        (bias_sigma **2 + self.bias_mu **2) /(2 * self.sigma_prior ** 2) - 0.5)

        return kldiv_weight + kldiv_bias
