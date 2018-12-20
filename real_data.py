#!/usr/bin/env python
# -*- coding: utf-8 -*-


#=========================================================================================================
#================================ 0. MODULE


# Computation
import numpy as np
import torch

from sklearn.metrics import log_loss

# Random
from torch.distributions.normal import Normal

# Perso
from common import *
from VL_BFGS import *
from feature_engineering import *


#=========================================================================================================
#================================ 1. DATA


SIZE = 250000

print('>> Loading data')
X = pd.read_parquet('data/data_train.csv')
y = pd.read_csv('data/labels_train.csv')

X = X[:SIZE]
y = y[:SIZE]

print('>> To tensor')
X = torch.tensor(X.values, dtype=torch.float)
y = torch.tensor(y.values, dtype=torch.float).squeeze()

print('>> Training on %2d samples and %2d features' % X.size())


#=========================================================================================================
#================================ 2. OPTIMIZATION


# Parameters
lbda = 0.1

# Initialization
w0 = torch.zeros(X.size(1))
f = logistic_loss
f_grad = logistic_grad


# CPU
def cpu(X, y, w0, lbda):
    optimizer = lbfgs(f, f_grad, m=10, vector_free=False, device='cpu')
    _, obj, time_comp, time_com = optimizer.fit(X, y, w0, lbda)

    return time_comp, time_com, obj[-1]


# GPU
def gpu(X, y, w0, lbda):
    optimizer = lbfgs(f, f_grad, m=10, vector_free=False, device='cuda:0')
    _, obj, time_comp, time_com = optimizer.fit(X, y, w0, lbda)

    return time_comp, time_com, obj[-1]


# GPU (vector free)
def gpu_VL(X, y, w0, lbda):
    optimizer = lbfgs(f, f_grad, m=10, vector_free=True, device='cuda:0')
    _, obj, time_comp, time_com = optimizer.fit(X, y, w0, lbda)

    return time_comp, time_com, obj[-1]


# Test
_, _, _ = gpu(X, y, w0, lbda)


##============================
## Monitoring time execution

REPEAT = 15

#==============
# CPU
cpu_time_comp = []
cpu_time_com = []
for _ in range(REPEAT):

    w0 = torch.rand(X.size(1))
    t1, t2, loss = cpu(X, y, w0, lbda)
    cpu_time_comp.append(t1)
    cpu_time_com.append(t2)

print('\n>> CPU computing time: %.2fs +-%.2fs' % (np.mean(cpu_time_comp), np.std(cpu_time_comp)))
print('>> CPU communication time: %.2fs +-%.2fs' % (np.mean(cpu_time_com), np.std(cpu_time_com)))
print('Reached logloss: %.3f\n' % loss)


#==============
# GPU
gpu_time_comp = []
gpu_time_com = []
for _ in range(REPEAT):

    w0 = torch.rand(X.size(1))
    t1, t2, loss = gpu(X, y, w0, lbda)
    gpu_time_comp.append(t1)
    gpu_time_com.append(t2)

print('>> GPU computing time: %.2fs +-%.2fs' % (np.mean(gpu_time_comp), np.std(gpu_time_comp)))
print('>> GPU communication time: %.2fs +-%.2fs' % (np.mean(gpu_time_com), np.std(gpu_time_com)))
print('Reached logloss: %.3f\n' % loss)


#==============
# GPU (vector free)
gpu_vl_time_comp = []
gpu_vl_time_com = []
for _ in range(REPEAT):

    w0 = torch.rand(X.size(1))
    t1, t2, loss = gpu_VL(X, y, w0, lbda)
    gpu_vl_time_comp.append(t1)
    gpu_vl_time_com.append(t2)

print('>> GPU (vector free) computing time: %.2fs +-%.2fs' % (np.mean(gpu_vl_time_comp), np.std(gpu_vl_time_comp)))
print('>> GPU (vector free) communicating time: %.2fs +-%.2fs' % (np.mean(gpu_vl_time_com), np.std(gpu_vl_time_com)))
print('Reached logloss: %.3f\n' % loss)
