#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.autograd import Variable
import logging
from pathlib import Path
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def to_numpy(var, gpu_used=False):
    return var.cpu().data.numpy().astype(np.float64) if gpu_used else var.data.numpy().astype(np.float64)

def to_tensor(ndarray, volatile=False, requires_grad=False, gpu_used=False, gpu_0 = 0):
    if gpu_used:
        return Variable(torch.from_numpy(ndarray).cuda(device=gpu_0).type(torch.cuda.DoubleTensor),
                        volatile=volatile,
                        requires_grad=requires_grad)
    else:
        return Variable(torch.from_numpy(ndarray).type(torch.DoubleTensor),
                        volatile=volatile,
                        requires_grad=requires_grad)

def soft_update(target, source, tau_update):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau_update) + param.data * tau_update
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)