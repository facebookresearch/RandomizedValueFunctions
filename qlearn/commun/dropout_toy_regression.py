# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from qlearn.commun.variational_dropout_layer import VariationalDropoutLinear

BAYES = True
use_cuda = True


class RegressionModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        #input_dim, output_dim, hidden_dim, n_hidden, n_flows_q, n_flows_r
        self.fc1 = VariationalDropoutLinear(1, 100, 50, 1, 2)
        self.fc2 = VariationalDropoutLinear(100, 1, 50, 1, 1)

    def forward(self, x):
        if self.training:
            x, kldiv1 = self.fc1.forward(x)
            x = F.relu(x)
            x, kldiv2 = self.fc2.forward(x)
            kldiv = kldiv1 + kldiv2
            return x, kldiv
        else:
            x = self.fc1.forward(x)
            x = F.relu(x)
            x = self.fc2.forward(x)
            return x

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()


class MLP(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

#X = torch.Tensor(20, 1).uniform_(-4, 4)
X = np.random.uniform(-4, 4, (20, 1)).astype('float32')
# X = np.random.rand(20, 1).astype('float32') * 8 - 4
sigma = 3
epsilon = np.random.normal(size=X.shape).astype('float32')
Y = np.power(X, 3) + sigma * epsilon

if BAYES:
    regressor = RegressionModel()
else:
    regressor = MLP()

x = Variable(torch.from_numpy(X))
y = Variable(torch.from_numpy(Y))
if use_cuda:
    x = x.cuda()
    y = y.cuda()
optimiser = optim.Adam(regressor.parameters(), lr=0.01)

if use_cuda:
    regressor.cuda()
    # y = y.cuda()

regressor.train()
for epoch in range(1000):
    regressor.zero_grad()
    if BAYES:
        regressor.reset_noise()
        # import pdb
        # pdb.set_trace()
        y_pred, kldiv = regressor(x)
        kl_reg = kldiv / 20.0
        mse = F.mse_loss(y_pred, y) / (2 * 9)
        loss = mse + kl_reg
    else:
        loss = F.mse_loss(regressor(x), y) / (2 * 9)
    loss.backward()
    optimiser.step()
    # if epoch % 10 == 0:
    if BAYES:
        print('epoch: {}, loss: {}, kl: {}, mse: {}'.format(epoch, loss.item(), kl_reg.item(), mse.item()))
    else:
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))

n_test = 500
x_test = np.linspace(-6, 6, n_test).reshape(n_test, 1).astype('float32')
y_preds = []

# regressor.eval()
# assert regressor.fc1.training == False

X_TEST = Variable(torch.from_numpy(x_test))
if use_cuda:
    X_TEST = X_TEST.cuda()
for _ in range(20):
    if BAYES:
        regressor.reset_noise()
        y_pred, _ = regressor(X_TEST)
        y_preds.append(y_pred.data.cpu().numpy())
    else:
        y_preds.append(regressor(X_TEST).data.cpu().numpy())
y_preds = np.array(y_preds).reshape(20, n_test)
y_preds_mean = np.mean(y_preds, axis=0)
y_preds_var = np.std(y_preds, axis=0)


plt.plot(x_test, y_preds_mean)
if BAYES:
    plt.fill_between(x_test.reshape(n_test,), y_preds_mean - 3 * y_preds_var, y_preds_mean + 3 * y_preds_var, alpha=0.5)
plt.plot(X, Y, 'x')
plt.ylim(-100, 100)
plt.savefig('toy_regression.png')
