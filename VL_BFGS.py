# !/usr/bin/env python
# -*- coding: utf-8 -*-



"""
Utils for the "Large-scale L-BFGS using MapReduce" project
"""



#=========================================================================================================
#================================ 0. MODULE


# Computation on CPU
import numpy as np

# Computation on GPU
import torch

# Utils
from time import time


#=========================================================================================================
#================================ 1. FUNCTIONS


def two_loops(grad_w, s_list, y_list):
    '''
    Parameters
    ----------
    grad_w (ndarray, shape [p,]) : current gradient
        
    s_list (list[]) : the past m values of s
    
    y_list (list[]) : the past m values of y
            
    Returns
    -------
    r (ndarray, shape [p, p]) : the L-BFGS direction
    '''
    q = grad_w.clone().cpu()
    alpha_list = []
    
    m = len(y_list)
    
    # First loop
    for i in reversed(range(m)):
        y = y_list[i]
        s = s_list[i]
        
        alpha_list.insert(0, s.matmul(q) / y.matmul(s))
        q -= alpha_list[0] * y
    
    if m != 0:
        y = y_list[-1]
        s = s_list[-1]

        q = (y.matmul(s) / y.matmul(y)) * q

    # Second loop
    for i in range(m):
        y = y_list[i]
        s = s_list[i]
        alpha = alpha_list[i]
        
        beta =  y.matmul(q) / y.matmul(s)
        q += (alpha - beta) * s
    
    return -q


def two_loops_vector_free(dot_matrix, b):
    '''
    Parameters
    ----------
    dot_matrix (ndarray, shape [2m + 1, 2m + 1]) : the precomputed dot product between all vectors

    b (ndarray, shape [2m + 1, n_features]) : all memory vectors and current gradient
            
    Returns
    -------
    r (ndarray, shape [p, p]) : the L-BFGS direction
    '''
    m = int((dot_matrix.size(0) - 1) / 2)

    alpha_list = []

    delta = torch.zeros((2*m + 1), dtype=torch.float).to(dot_matrix.device)
    delta[2*m] = -1

    for i in reversed(range(m)):
        denom = dot_matrix[i, m + i]
        num = torch.sum(delta * dot_matrix[i, :])

        alpha_list.insert(0, num / denom)
        delta[m + i] -= alpha_list[0]

    for i in range(2*m + 1):
        delta[i] *= dot_matrix[m - 1, 2*m -1] / dot_matrix[2*m - 1, 2*m - 1]

    for i in range(m):
        denom = dot_matrix[i, m + i]
        num = torch.sum(delta * dot_matrix[m + i, :])

        beta = num / denom

        delta[i] += alpha_list[i] - beta

    for i in range(2*m + 1):
        b[i, :] *= delta[i]

    direction = b.sum(dim=0)

    return direction


def dot_product(y_list, s_list, grad_w):
    """    
    Parameters
    ----------
    y_list (list of m array of size n_features) : memory vectors y's

    s_list (list of m array of size n_features) : memory vectors s's
       
    grad_w (ndarray, shape [n_features]) : current gradient

    Returns
    -------
    dot_matrix (ndarray, shape [2m + 1, 2m + 1]) : the precomputed dot product between all vectors

    b (ndarray, shape [2m + 1, n_features]) : all memory vectors and current gradient
    """
    m = len(y_list)
    n_features = grad_w.size(0)

    # Build matrix of all vectors
    b = torch.empty((2*m + 1, n_features), dtype=torch.float).to(grad_w.device)

    for i, tensor in enumerate(s_list):
        b[i, :] = tensor

    for i, tensor in enumerate(y_list):
        b[m + i, :] = tensor

    b[2 * m, :] = grad_w

    # Dot product between all vectors
    dot_matrix = b.matmul(b.transpose(0, 1))

    return dot_matrix, b



def line_search(f, f_grad, c1, c2, current_f, current_grad, direction, X, y, lbda, w):
    """
    Find the best gradient descent step using the Armijo and Wolfe condition
    """
    alpha = 0
    beta = 'inf'
    step = 1

    for i in range(10):

        next_f = f(X, y, lbda, w.add(step * direction)).item()
        f1 = (current_f + c1 * step * current_grad.matmul(direction)).item()

        next_grad = f_grad(X, y, lbda, w.add(step * direction))
        f2 = next_grad.matmul(direction).item()
        f3 = (c2 * current_grad.matmul(direction)).item()

        """
        (the method ".item()" bring back the scalars on CPU)

        Here the computation takes places on the CPU because 
        GPUs are slower when it comes to conditions (if statement)
        """

        if next_f > f1:                                # Armijo condition
            beta = step
            step = (alpha + beta) / 2
        elif f2 < f3:                                  # Wolfe condition 
            alpha = step
            if beta == 'inf':
                step = 2 * alpha
            else:
                step = (alpha + beta) / 2
        else:
            break

    """
    Since the step has already been done, we return the next value of the loss function
    and the next value of gradient so as to prevent from recomputing it
    """

    return step, next_f, next_grad 



#=========================================================================================================
#================================ 2. ALGORITHM


class lbfgs(object):

    def __init__(self, f, f_grad, m=10, vector_free=False, device='cpu'):
        
        self.c1 = 0.0001
        self.c2 = 0.9
        self.max_iter = 20

        self.m = m

        self.all_f = []

        self.f = f
        self.f_grad = f_grad

        self.device = device
        self.vector_free = vector_free


    def fit(self, X, target, w0, lbda):

        t0 = time()

        #========================================
        # Moving data to computing device

        X = X.to(self.device)
        target = target.to(self.device)
        w = w0.to(self.device)

        t1 = time()

        #========================================
        # Computing first value of the objective function

        new_f = self.f(X, target, lbda, w).item()
        self.all_f.append(new_f)

        #========================================
        # Computing first gradient

        grad_w = self.f_grad(X, target, lbda, w)

        #========================================
        # Creating memory lists

        y_list = []
        s_list = []

        for k in range(self.max_iter):
            

            #========================================
            # Compute the search direction

            """
            Both two_loop functions are computed on the CPU
            because they outperform GPUs when it comes to
            small sequential computations with numerous for-loops.
            """

            if self.vector_free:
                dot_matrix, b = dot_product(y_list, s_list, grad_w)

                dot_matrix = dot_matrix.cpu()
                b = b.cpu()

                d = two_loops_vector_free(dot_matrix, b)
                d = d.to(self.device)

            else:
                d = two_loops(grad_w, s_list, y_list)
                d = d.to(self.device)


            #========================================
            # Compute the step size using line search

            step, new_f, new_grad  = line_search(self.f, self.f_grad, self.c1, self.c2,
                                                   new_f, grad_w, d, X, target, lbda, w)


            #========================================
            # Compute the new value of w

            s = step * d
            w = w.add(s)


            #========================================
            # Compute y

            y = new_grad.add(- grad_w)
            

            #========================================
            # Update the memory
            
            """
            The memory vectors are direcly stored on the CPU since the 
            for-loops take place there
            """
            y_list.append(y.cpu())
            s_list.append(s.cpu())

            if len(y_list) > self.m:
                y_list.pop(0)
                s_list.pop(0)
                

            #========================================
            # Monitoring
            
            self.all_f.append(new_f)

            l_inf_norm_grad = torch.max(torch.abs(new_grad)).item()

            if l_inf_norm_grad < 1e-5:
                break
                
            grad_w = new_grad

        computing_time = time() - t1
        communication_time = t1 - t0
        
        return w.cpu().numpy(), np.array(self.all_f), computing_time, communication_time