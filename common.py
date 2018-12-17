# !/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Utils for the "Large-scale L-BFGS using MapReduce" project
"""


#=========================================================================================================
#================================ 0. MODULE


# Computation on CPU
import numpy as np
from numpy.linalg import norm

# Computation on GPU
import torch
from torch.distributions.normal import Normal


#=========================================================================================================
#================================ 1. FUNCTIONS

#===========================================================
# Simulation of data

def simulate_data(coefs, n_samples=1000, for_logreg=False, device='cpu'):
    '''
    Simulate data for regression problems
    '''
    n_features = len(coefs)
    
    n = Normal(0, 1)

    X = torch.empty(n_samples, n_features)

    for var in range(n_features):
        X[:, var] = n.sample(torch.Size([n_samples]))

    noise = n.sample(torch.Size([n_samples]))
    target = X.matmul(coefs) + noise

    if for_logreg:
        target = torch.sign(target)

    return X, target


#===========================================================
# Loss functions and their gradient

def linear_grad(X, y, lbda, w):
    n = X.size(0)
    return (-1. / n) * X.transpose(0, 1).matmul(y - X.matmul(w)) + lbda * w


def linear_loss(X, y, lbda, w):
    n = X.size(0)
    A = X.matmul(w) - y
    return A.matmul(A) / (2. * n) + lbda * w.matmul(w) / 2.


def logistic_grad(X, y, lbda, w):
    n = X.size(0)
    return (1. / n) * X.transpose(0,1).matmul( - y / (torch.exp(y * X.matmul(w)) + 1.)) + lbda * w


def logistic_loss(X, y, lbda, w):
    n = X.size(0)
    bAx = y * X.matmul(w)
    return torch.mean(torch.log(1. + torch.exp(- bAx))) + lbda * w.matmul(w) / 2.


#===========================================================
# Numpy version

def linear_grad_numpy(w, X, y, lbda):
    n = X.shape[0]
    return (- 1. / n) * X.T.dot(y - X.dot(w)) + lbda * w


def linear_loss_numpy(w, X, y, lbda):
    n = X.shape[0]
    return norm(X.dot(w) - y) ** 2 / (2. * n) + lbda * norm(w) ** 2 / 2.


def logistic_grad_numpy(w, X, y, lbda):
    n = X.shape[0]
    return (1 / n) * X.T.dot( - y / (np.exp(y * X.dot(w)) + 1.)) + lbda * w


def logistic_loss_numpy(w, X, y, lbda):
    n = X.shape[0]
    bAx = y * np.dot(X, w)
    return np.mean(np.log(1. + np.exp(- bAx))) + lbda * norm(w) ** 2 / 2.