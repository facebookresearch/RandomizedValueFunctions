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

from qlearn.commun.mnf_layer import MNFLinear

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
        self.fc1 = MNFLinear(1, 100, 16, 0, 2, 2)
        self.fc2 = MNFLinear(100, 1, 16, 0, 2, 2)

    def forward(self, x, kl=True):
        if self.training:
            if kl:
                x, kldiv1 = self.fc1.forward(x, kl=True)
                x = F.relu(x)
                x, kldiv2 = self.fc2.forward(x, kl=True)
                kldiv = kldiv1 + kldiv2
                return x, kldiv
            else:
                x = self.fc1.forward(x, kl=False)
                x = F.relu(x)
                x = self.fc2.forward(x, kl=False)
                return x
        else:
            x = self.fc1.forward(x, kl=False)
            x = F.relu(x)
            x = self.fc2.forward(x, kl=False)
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

if __name__ == '__main__':

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
    if BAYES:
        regressor.reset_noise()
    for epoch in range(1000):
        regressor.zero_grad()
        if BAYES:
            regressor.reset_noise()
            y_pred, kldiv = regressor(x, kl=True)
            kl_reg = kldiv / 20.0
            # y_pred = regressor(x, kl=False)
            mse = F.mse_loss(y_pred, y) / (2 * 9)
            loss = mse + kl_reg
            # loss = mse
        else:
            loss = F.mse_loss(regressor(x), y) / (2 * 9)
        loss.backward()
        optimiser.step()
        if epoch % 10 == 0:
            if BAYES:
                print('epoch: {}, loss: {}, kl: {}, mse: {}'.format(epoch, loss.item(), kl_reg.item(), mse.item()))
                # print('epoch: {}, loss: {}'.format(epoch, loss.item()))
            else:
                print('epoch: {}, loss: {}'.format(epoch, loss.item()))

    n_test = 500
    x_test = np.linspace(-6, 6, n_test).reshape(n_test, 1).astype('float32')
    y_preds = []

    regressor.train()
    # assert regressor.fc1.training == False

    X_TEST = Variable(torch.from_numpy(x_test))
    if use_cuda:
        X_TEST = X_TEST.cuda()
    for _ in range(20):
        if BAYES:
            regressor.reset_noise()
            y_pred = regressor(X_TEST, kl=False)
            # y_pred = regressor(X_TEST)
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
    plt.savefig('global_mnf_toy_regression.png')
    # plt.show()