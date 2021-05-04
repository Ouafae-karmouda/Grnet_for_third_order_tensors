# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:56:34 2021

@author: adminlocal
"""
import torch.nn
import numpy as np
from torch.autograd import Function
from tlinalg import t_product, t_transpose
tenType = torch.FloatTensor

m0 =16


class Reorth(Function):
        
        def __init__(self, inputs):
            [batch_size, d1, m0, q] = inputs.shape
            self.Q_fft = torch.zeros_like(inputs).type(tenType).cuda()
            self.R_fft =  torch.zeros(batch_size, q, m0, q).cuda()
            self.m0 = m0
        @staticmethod
        def forward(self, inputs):
            """
            inputs: batch of tensors X1 from the FRMap layer of size (d1, m0, q)
            m0 : number of filters
            d1 > q : to have a reduced form of QR
            """
            inputs = inputs.type(tenType)
            [batch_size, d1, m0, q] = inputs.shape
            inputs_fft = np.fft.fft(inputs.detach().numpy())
            inputs_fft = torch.from_numpy(inputs_fft)
            Q_tensor =  torch.zeros_like(inputs).type(tenType).cuda()
            R_tensor =  torch.zeros(batch_size, q, m0, q).cuda()
            #tronquer à la dimension de filtrage
            """
            for q0 in range(q):
                (Q, R) = torch.qr(inputs_fft[:,:, :, q0], some=True)
                Q_tensor[:,:,:,q0] = Q
                R_tensor[:,:,:,q0] = R
            """
            #tronquer à la 3e dimension des tenseurs
            for m in range(m0):
                (Q, R) = torch.qr(inputs_fft[:,:, m, :], some=True)
                Q_tensor[:,:,m,:] = Q
                R_tensor[:,:,m,:] = R
            self.Q_fft = Q_tensor
            self.R_fft = R_tensor
            Q_ifft_tensor = np.fft.ifft(Q_tensor.cpu()).real
            return  torch.from_numpy(Q_ifft_tensor).cuda()
        
        @staticmethod
        def backward(self, grad_outputs):
            """
            grad_outputs : gradients of the next layer which is the ProjMap of size (batch_size, d1, d1, q)
    
            """
            [batch_size, d1, d1, q] = grad_outputs.shape
            
            d1 = 30
            dLdQ = grad_outputs.type(tenType).cuda()
            grad = torch.zeros_like(grad_outputs).type(tenType).cuda() #grad and gradout same dims
        
        
            for m in range(m0): 
                print("m", m)
                print("Q_fft shape", self.Q_fft.shape)
                Q = self.Q_fft[:,:,m,:]
                R = self.R_fft[:,:,m,:]
                R_inv = torch.pinverse(R)
                print("Q.shape", Q.shape)
                print("eye(d1.shape", torch.eye(d1).cuda().shape )
                list_eye = [torch.eye(d1).cuda() for i in range(batch_size)]
                tensors_eye = torch.stack(list_eye)
                S = tensors_eye - torch.matmul(Q, Q.transpose(1,2) )
                print("S.shape", S.shape)
                ele_1 = torch.matmul(S.transpose(1,2), dLdQ[:, :, m, :])
                temp = torch.matmul(Q.transpose(1,2), dLdQ[:, :, m, :])
                temp_bsym = torch.tril(temp) - torch.tril(temp.transpose(1, 2))
                ele_2 = torch.matmul(Q, temp_bsym)
                grad[:, :,m, :] = torch.matmul(ele_1+ele_2, R_inv.transpose(1,2))
            
            grad_ifft = np.fft.ifft(grad.cpu()).real
           
            return torch.from_numpy(grad_ifft).cuda()
        
        
class ProjPooling(Function):
    
    @staticmethod
    def forward(inputs):
        """
        inputs : batch of tensors X2 of size (d1, d1, q)
        """
        
        inputs = inputs.type(tenType)
        [batch_size, d1, d1, q] = inputs.shape
          
        return t_product(inputs, t_transpose(inputs))
            
        
def call_ProjPooling(inputs):
    return ProjPooling().apply(inputs)
        
def call_reorthmap(inputs):
    reorth = Reorth(inputs)
    return reorth.apply(inputs)

    
def reim_grad_fct(W, euc_grad):
    """
    reim_grad = euc_grad - euc_grad * W^T * W
    """
    W_t = torch.transpose(W, 0, 1)
    temp = torch.matmul(W_t, W)
    reim_grad = euc_grad - torch.matmul(euc_grad, temp)
    #reim_grad = retraction(W - lamda* grad_tilde)
    
    return reim_grad
    
def update_params(lr, W, euc_grad):
    """
    W_up: update of W_up = retraction(W - lr*reim_grad)
    """
    reim_grad = reim_grad_fct(W, euc_grad)
    W_up = retraction(W - lr * reim_grad)
    
    return W_up


def retraction(M):
    
    """
    projection of M onto the psd manifold
    
    """
    eig_val, eig_vectors = np.linalg.eig(M)
    abs_eig_val = abs(eig_val)
    
    proj_M = np.dot(eig_vectors, np.dot(np.diag(abs_eig_val), eig_vectors.transpose()))
    
    return proj_M


"""
 #tronquer à la dimension de filtrage
            
            for q0 in range(q):
                (Q, R) = torch.qr(inputs_fft[:,:, :, q0], some=True)
                Q_tensor[:,:,:,q0] = Q
                R_tensor[:,:,:,q0] = R
"""