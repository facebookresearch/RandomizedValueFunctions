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
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from qlearn.commun.bayes_backprop_layer import BayesBackpropLinear

BAYES = True
use_cuda = False
seed = 60

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class RegressionModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        #input_dim, output_dim, hidden_dim, n_hidden, n_flows_q, n_flows_r
        self.fc1 = BayesBackpropLinear(1, 100)
        self.fc2 = BayesBackpropLinear(100, 1)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = F.relu(x)
        x = self.fc2.forward(x)
        return x

    def get_reg(self):
        reg = self.fc1.kldiv()
        reg += self.fc2.kldiv()
        return reg

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
        kl_reg = 1.0 * regressor.get_reg()/x.size()[0]
        regressor.reset_noise()
        mse = F.mse_loss(regressor(x), y) / (2 * 9)
        loss = mse + kl_reg
    else:
        loss = F.mse_loss(regressor(x), y) / (2 * 9)
    loss.backward()
    optimiser.step()
    if epoch % 10 == 0:
        if BAYES:
            print('epoch: {}, loss: {}, kl: {}, mse: {}'.format(epoch, loss.item(), kl_reg.item(), mse.item()))
        else:
            print('epoch: {}, loss: {}'.format(epoch, loss.item()))

x_test = np.linspace(-6, 6, 100).reshape(100, 1).astype('float32')
y_preds = []

# regressor.eval()
# assert regressor.fc1.training == False

X_TEST = Variable(torch.from_numpy(x_test))
if use_cuda:
    X_TEST = X_TEST.cuda()
for _ in range(20):
    if BAYES:
        regressor.reset_noise()
        y_preds.append(regressor(X_TEST).data.cpu().numpy())
    else:
        y_preds.append(regressor(X_TEST).data.cpu().numpy())
y_preds = np.array(y_preds).reshape(20, 100)
y_preds_mean = np.mean(y_preds, axis=0)
y_preds_var = np.std(y_preds, axis=0)


plt.plot(x_test, y_preds_mean)
if BAYES:
    plt.fill_between(x_test.reshape(100,), y_preds_mean - 3 * y_preds_var, y_preds_mean + 3 * y_preds_var, alpha=0.5)
plt.plot(X, Y, 'x')
plt.ylim(-100, 100)
plt.savefig('bb_toy_regression.png')
