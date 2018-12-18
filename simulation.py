#!/usr/bin/env python
# -*- coding: utf-8 -*-


#=========================================================================================================
#================================ 0. MODULE


# Computation
import numpy as np
import torch

# Random
from torch.distributions.normal import Normal

# Perso
from common import *
from VL_BFGS import *


#=========================================================================================================
#================================ 1. DATA


def simulate():

    n = Normal(0, 1)
    coefs = n.sample(torch.Size([n_features]))

    return simulate_data(coefs, n_samples, for_logreg=True)


#=========================================================================================================
#================================ 2. OPTIMIZATION


# Parameters
n_features = 5000
n_samples = 100000

lbda = 0.1


# Initialization
w0 = torch.zeros(n_features)
f = logistic_loss
f_grad = logistic_grad


# CPU
def cpu(X, y, w0, lbda):
    optimizer = lbfgs(f, f_grad, m=10, vector_free=False, device='cpu')
    _, _, cpu_time_comp, cpu_time_com = optimizer.fit(X, y, w0, lbda)

    return cpu_time_comp, cpu_time_com


# GPU
def gpu(X, y, w0, lbda):
    optimizer = lbfgs(f, f_grad, m=10, vector_free=False, device='cuda:0')
    _, _, gpu_time_comp, gpu_time_com = optimizer.fit(X, y, w0, lbda)

    return gpu_time_comp, gpu_time_com


# GPU (vector free)
def gpu_VL(X, y, w0, lbda):
    optimizer = lbfgs(f, f_grad, m=10, vector_free=True, device='cuda:0')
    _, _, gpu_vl_time_comp, gpu_vl_time_com = optimizer.fit(X, y, w0, lbda)

    return gpu_vl_time_comp, gpu_vl_time_com


# Test
X, y = simulate()
t1, t2 = gpu(X, y, w0, lbda)


##============================
## Monitoring time execution

REPEAT = 20
SHUFFLE = False

#==============
# CPU
cpu_time_comp = []
cpu_time_com = []
for _ in range(REPEAT):

    if SHUFFLE:
        X, y = simulate()
    t1, t2 = cpu(X, y, w0, lbda)
    cpu_time_comp.append(t1)
    cpu_time_com.append(t2)

print('\n>> CPU computing time: %.2fs +-%.2fs' % (np.mean(cpu_time_comp), np.std(cpu_time_comp)))
print('>> CPU communication time: %.2fs +-%.2fs\n' % (np.mean(cpu_time_com), np.std(cpu_time_com)))


#==============
# GPU
gpu_time_comp = []
gpu_time_com = []
for _ in range(REPEAT):

    if SHUFFLE:
        X, y = simulate()
    t1, t2 = gpu(X, y, w0, lbda)
    gpu_time_comp.append(t1)
    gpu_time_com.append(t2)

print('>> GPU computing time: %.2fs +-%.2fs' % (np.mean(gpu_time_comp), np.std(gpu_time_comp)))
print('>> GPU communication time: %.2fs +-%.2fs\n' % (np.mean(gpu_time_com), np.std(gpu_time_com)))


#==============
# GPU (vector free)
gpu_vl_time_comp = []
gpu_vl_time_com = []
for _ in range(REPEAT):

    if SHUFFLE:
        X, y = simulate()
    t1, t2 = gpu_VL(X, y, w0, lbda)
    gpu_vl_time_comp.append(t1)
    gpu_vl_time_com.append(t2)

print('>> GPU (vector free) computing time: %.2fs +-%.2fs' % (np.mean(gpu_vl_time_comp), np.std(gpu_vl_time_comp)))
print('>> GPU (vector free) communicating time: %.2fs +-%.2fs\n' % (np.mean(gpu_vl_time_com), np.std(gpu_vl_time_com)))