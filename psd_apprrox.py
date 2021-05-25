# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:47:21 2021

@author: adminlocal
"""

import numpy as np
import torch

def FixedRankPsdApp(A,r,k, mu):
    """
    Returns a fixed rank approximation of the psd matrix A
    """
    n = A.shape[0]
   
    norm = torch.norm(A)
    nu = mu*norm
    #WE can orthogonalize Omega
    Omega = torch.randn(n,k)
    Q, R = torch.qr(Omega)
    Omega = Q
    Y = torch.matmul(A, Omega)
    Ynu = Y + nu*Omega
    B = torch.matmul(torch.conj(Omega).transpose(1,0), Ynu)
    temp_B = (B+ torch.conj(B).transpose(1,0))/2
    #print(temp_B - temp_B.transpose(1,0))
    C = torch.cholesky(temp_B)
    #print(torch.matmul(C, torch.conj(C)) == temp_B)
    E = torch.matmul(Ynu, torch.inverse(C))
    U, S, V = torch.svd(E)
    #print(torch.maximum(torch.diag(S**2) - nu*torch.eye(r), torch.zeros(r,r)))
    temp = torch.maximum(torch.diag(S**2) - nu*torch.eye(r), torch.zeros(r,r))
    A_hat = torch.matmul(U, torch.matmul(temp, torch.conj(U).transpose(1,0)))
    return A_hat
    
mu = 2.2 * 10**(-16)
M = torch.randn(3,4)
r = np.linalg.matrix_rank(M)
A = torch.matmul(M, torch.conj(M).transpose(1,0))
A_hat = FixedRankPsdApp(A,r,r, mu)