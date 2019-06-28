# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from qlearn.commun.norm_flows import MaskedNVPFlow


class MNFLinear(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, n_hidden, n_flows_q, n_flows_r,
                 prior_var=1.0, threshold_var=0.5):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.n_flows_q = n_flows_q
        self.n_flows_r = n_flows_r
        self.prior_var = prior_var
        self.threshold_var = threshold_var

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logstd = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

        self.qzero_mu = nn.Parameter(torch.Tensor(in_features))
        self.qzero_logvar = nn.Parameter(torch.Tensor(in_features))
        # auxiliary variable c, b1 and b2 are defined in equation (9) and (10)
        self.rzero_c = nn.Parameter(torch.Tensor(in_features))
        self.rzero_b1 = nn.Parameter(torch.Tensor(in_features))
        self.rzero_b2 = nn.Parameter(torch.Tensor(in_features))

        self.flow_q = MaskedNVPFlow(in_features, hidden_dim, n_hidden, n_flows_q)
        self.flow_r = MaskedNVPFlow(in_features, hidden_dim, n_hidden, n_flows_r)

        self.register_buffer('epsilon_z', torch.Tensor(in_features))
        self.register_buffer('epsilon_weight', torch.Tensor(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_noise(self):
        epsilon_z = torch.randn(self.in_features)
        epsilon_weight = torch.randn(self.out_features, self.in_features)
        epsilon_bias = torch.randn(self.out_features)
        self.epsilon_z.copy_(epsilon_z)
        self.epsilon_weight.copy_(epsilon_weight)
        self.epsilon_bias.copy_(epsilon_bias)
        self.flow_q.reset_noise()
        self.flow_r.reset_noise()

    def reset_parameters(self):

        in_stdv = np.sqrt(4.0 / self.in_features)
        out_stdv = np.sqrt(4.0 / self.out_features)
        stdv2 = np.sqrt(4.0 / (self.in_features + self.out_features))

        self.weight_mu.data.normal_(0, stdv2)
        self.weight_logstd.data.normal_(-9, 1e-3 * stdv2)
        self.bias_mu.data.zero_()
        self.bias_logvar.data.normal_(-9, 1e-3 * out_stdv)

        self.qzero_mu.data.normal_(1 if self.n_flows_q == 0 else 0, in_stdv)
        self.qzero_logvar.data.normal_(np.log(0.1), 1e-3 * in_stdv)
        self.rzero_c.data.normal_(0, in_stdv)
        self.rzero_b1.data.normal_(0, in_stdv)
        self.rzero_b2.data.normal_(0, in_stdv)

    def sample_z(self, kl=True):
        if self.training:
            qzero_std = torch.exp(0.5 * self.qzero_logvar)
            z =  self.qzero_mu + qzero_std * self.epsilon_z
        else:
            z = self.qzero_mu
        if kl:
            z, logdets = self.flow_q(z, kl=True)
            return z, logdets
        else:
            z = self.flow_q(z, kl=False)
            return z

    def forward(self, input, kl=True):
        if self.training:
            if kl:
                z, logdets = self.sample_z(kl=True)
            else:
                z = self.sample_z(kl=False)
            weight_std = torch.clamp(torch.exp(self.weight_logstd), 0, self.threshold_var)
            bias_std = torch.clamp(torch.exp(0.5 * self.bias_logvar), 0, self.threshold_var)
            weight_mu = z.view(1, -1) * self.weight_mu
            weight = weight_mu + weight_std * self.epsilon_weight
            bias = self.bias_mu + bias_std * self.epsilon_bias
            out = F.linear(input, weight, bias)
            if not kl:
                return out
            else:
                kldiv_weight = 0.5 * (- 2 * self.weight_logstd + torch.exp(2 * self.weight_logstd)
                                      + weight_mu * weight_mu - 1).sum()
                kldiv_bias = 0.5 * (- self.bias_logvar + torch.exp(self.bias_logvar)
                                    + self.bias_mu * self.bias_mu - 1).sum()
                logq = - 0.5 * self.qzero_logvar.sum()
                logq -= logdets

                cw = F.tanh(torch.matmul(self.rzero_c, weight.t()))

                mu_tilde = torch.mean(self.rzero_b1.ger(cw), dim=1)
                neg_log_var_tilde = torch.mean(self.rzero_b2.ger(cw), dim=1)

                z, logr = self.flow_r(z)

                z_mu_square = (z - mu_tilde) * (z - mu_tilde)
                logr += 0.5 * (- torch.exp(neg_log_var_tilde) * z_mu_square
                               + neg_log_var_tilde).sum()


                kldiv = kldiv_weight + kldiv_bias + logq - logr
                return out, kldiv
        else:
            assert kl == False
            z = self.sample_z(kl=False)
            weight_mu = z.view(1, -1) * self.weight_mu
            out = F.linear(input, weight_mu, self.bias_mu)
            return out
