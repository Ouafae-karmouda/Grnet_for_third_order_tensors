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
        
        @staticmethod
        def forward(ctx, inputs):
            """
            inputs: batch of tensors X1 from the FRMap layer of size (d1, m0, q)
            m0 : number of filters
            d1 > q : to have a reduced form of QR
            """
            [batch_size, d1, m0, q] = inputs.shape
            ctx.m0 = m0            
            print("Forward Reorth")
            inputs = inputs.type(tenType)
            [batch_size, d1, m0, q] = inputs.shape
            inputs_fft = torch.fft.fft(inputs)

            list_Q = []
            list_R = []
            for q0 in range(q):
                (Q, R) = torch.qr(inputs_fft[:,:, :, q0], some=True)
                list_Q.append(Q)
                list_R.append(R)

            ctx.Q_fft = torch.stack(list_Q).view(batch_size, d1, m0, q).type(torch.FloatTensor).cuda()
            ctx.R_fft = torch.stack(list_R).view(batch_size, m0, m0, q).type(torch.FloatTensor).cuda()
            return  torch.real(torch.fft.ifft(ctx.Q_fft))
        
        @staticmethod
        def backward(ctx, grad_outputs):
            """
            grad_outputs : gradients of the next layer which is the ProjMap of size (batch_size, d1, d1, q)
    
            """
            print("Backward Reorth")
            print(grad_outputs.shape)
            [batch_size, d1, d2, q] = grad_outputs.shape
            print("d1 reorth",d1)
            #d1 = 28
            dLdQ = grad_outputs.type(tenType).cuda()
            #grad = torch.zeros_like(grad_outputs).type(tenType).cuda() #grad and gradout same dims
            #grad = torch.zeros(batch_size, d1, m0, q).cuda()
            list_grad =[]
            for q0 in range(q): 
                #print("m", m)
                #print("Q_fft shape", ctx.Q_fft.shape)
                Q = ctx.Q_fft[:,:,:,q0]
                R = ctx.R_fft[:,:,:,q0]
                R_inv = torch.pinverse(R)
                list_eye = [torch.eye(d1).cuda() for i in range(batch_size)]
                tensors_eye = torch.stack(list_eye)
                S = tensors_eye - torch.matmul(Q, Q.transpose(1,2) )
                S = S.type(torch.FloatTensor).cuda()
                #print("Q.dtype", Q.dtype)
                #print("dldQ.dtype", dLdQ.dtype)
                ele_1 = torch.matmul(S.transpose(1,2), dLdQ[:, :, :, q0])
                temp = torch.matmul(Q.transpose(1,2), dLdQ[:, :, :, q0])
                temp_bsym = torch.tril(temp) - torch.tril(temp.transpose(1, 2))
                ele_2 = torch.matmul(Q, temp_bsym)
                list_grad.append(torch.matmul(ele_1+ele_2, R_inv.transpose(1,2)))
                
            grad =  torch.stack(list_grad).view(batch_size, d1, d2, q).cuda()
                
            return  torch.real(torch.fft.ifft(grad))
        

class OrthMap(Function):

    @staticmethod
    def forward(ctx, input):
        """
        inputs batch of tensors from the ProjPooling of size (d1, d1, q_tilde)
        m1 : the nbr of matrices to keep from the U matrix of t-SVD 
        """
        m1 =10
        print("Forward OrthMap")
        
        [batch_size, d1, d1, q_tilde] = input.shape
        list_Ufft=[]
        inputs_fft = torch.fft.fft(input).cuda()
        list_S = []
        for i in range(q_tilde):
            (U, S, V) = torch.svd(inputs_fft[:,:, :, i], some=True)
            list_Ufft.append(U[:,:,:])
            list_S.append(S)
        ctx.U_fft_tensor = torch.stack(list_Ufft).view(batch_size, d1,d1,q_tilde).type(tenType).cuda()
        ctx.S_fft_tensor = torch.stack(list_S)

        return  torch.real(torch.fft.ifft(ctx.U_fft_tensor[:,:,:m1,:]))
    
    
    @staticmethod
    def backward(ctx, grad_outputs):
        """
        grad_outputs is of shape (batch_size,d1, m1,q_tilde)
        m1 : nbr of matrices retained after a truncated t-SVD
        q_tilde: new size after Pooling 
        """
        print("Backward OrthMap")
        print(grad_outputs.shape)
        [batch_size, d1, m1, q_tilde] = grad_outputs.shape
        dLdU = grad_outputs.type(tenType).cuda() #needs concatenation of zeros to complete [bs, d1, d1]
        list_grad = []
        for i in range(q_tilde):
             #Ks = torch.zeros((dLdU.shape[0], dLdU.shape[1], dLdU.shape[1])).type(tenType)
                 ctx.S_fft_tensor = ctx.S_fft_tensor.contiguous()
                 vs_1 = ctx.S_fft_tensor[i].view( ctx.S_fft_tensor.shape[2],  ctx.S_fft_tensor.shape[1],1)
                 #print('vs_1.shape', vs_1.shape)
                 vs_2 = ctx.S_fft_tensor[i].view(1,ctx.S_fft_tensor.shape[1], ctx.S_fft_tensor.shape[2])
                 #print('vs_2.shape', vs_2.shape)
                 K = 1.0 / (vs_1 - vs_2).cuda() # matrice P
                 #print("K.shape", K.shape)
                 # K.masked_fill(mask_diag, 0.0)
                 K[K >= float("inf")] = 0.0
                 #K = K.transpose(2,1)
 
                 U_fft_tensor_t = ctx.U_fft_tensor.transpose(2,1)
                 #print(" dLdU[:,:,:,i].shape",  dLdU[:,:,:,i].shape)
                 #print("U_fft_tensor_t[:,:,:,i]", U_fft_tensor_t[:,:,:,i].shape)
                 temp = torch.matmul(U_fft_tensor_t[:,:,:,i] , dLdU[:,:,:,i]).type(tenType).cuda()
                 #print("temp.shape", temp.shape)
                 #print("torch.zeros((batch_size, d1, d1-dLdU[:,:,:,i].shape[-1])", torch.zeros((batch_size, d1, d1-dLdU[:,:,:,i].shape[-1])).shape)
                 temp = torch.cat((temp, torch.zeros((batch_size, d1, d1-dLdU[:,:,:,i].shape[-1]), dtype=torch.float32).cuda()), dim=2)
                 K = K.transpose(1,0)
                 temp = K*temp
                 temp = temp.type('torch.cuda.FloatTensor')
                 temp = torch.matmul(ctx.U_fft_tensor[:,:,:,i], temp)
                 
                 temp = torch.matmul(temp, U_fft_tensor_t[:,:,:,i] )
                 list_grad.append(temp)
        grad = torch.stack(list_grad).view(dLdU.shape[0], dLdU.shape[1], d1, q_tilde)     
        return torch.real( torch.fft.ifft(grad))
         

            
        
    
    
def call_ProjPooling(inputs, kernel_size , strides):
    """
    inputs of size (batch_size, d1, d1, q) & return a batch of tenosrs of size (batch_size, d1, d1, q_prime)
    where q_prime = q/pool_size
    """
    m = torch.nn.MaxPool3d(kernel_size, strides)
    outputs = m(inputs)
    return outputs
    

def call_ProjMap(inputs):
        """
        inputs : batch of tensors X2 of size (d1, m0, q)
        """
        
        inputs = inputs.type(tenType)
        #[batch_size, d1, m0, q] = inputs.shape
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            outputs.append(torch.from_numpy(t_product(inputs[i,:,:,:], t_transpose(inputs[i,:,:,:]) )))
        outputs = torch.stack(outputs).cuda()
        return outputs

def call_reorthmap(inputs):
    #reorth = Reorth(inputs)
    return Reorth(inputs).apply(inputs)

def call_orthmap(inputs):
    #orth = OrthMap(inputs)
    return  OrthMap(inputs).apply(inputs)
    
def reim_grad_fct(W, euc_grad):
    """
    reim_grad = euc_grad - euc_grad * W^T * W
    """
    #print("W", W.shape)
    W_t = W.transpose(1, 0)
    temp = torch.matmul(W_t, W)
    reim_grad = euc_grad - torch.matmul(euc_grad, temp)
    #reim_grad = retraction(W - lamda* grad_tilde)
    
    return reim_grad
    
def update_params_filters(W, euc_grad,lr):
    """
    W_up: update of W_up = retraction(W - lr*reim_grad)
    """
    #print('compute reim gradients')
    reim_grad = reim_grad_fct(W, euc_grad)
    P = torch.matmul(W.transpose(1,0), W)
    #W_up = retraction(W - lr * reim_grad)
    P_ret, abs_eig_val, eig_vectors = retraction(P - lr * reim_grad)
    W_up = torch.matmul(torch.sqrt(abs_eig_val), eig_vectors.transpose(1,0))
    return W_up

def retraction(M):
    
    """
    projection of M onto the psd manifold
    
    """
    #eig_val, eig_vectors = np.linalg.eig(M.cpu().detach().numpy())
    #print(M)
    eig_val, eig_vectors = torch.eig(M, eigenvectors = True)
    abs_eig_val = abs(eig_val[:,1])
    
    proj_M = torch.matmul(eig_vectors, torch.matmul(torch.diag(abs_eig_val), eig_vectors.transpose(1,0)))
    
    return proj_M.cuda(), abs_eig_val, eig_vectors


"""
 #tronquer à la dimension de filtrage
            
            for q0 in range(q):
                (Q, R) = torch.qr(inputs_fft[:,:, :, q0], some=True)
                Q_tensor[:,:,:,q0] = Q
                R_tensor[:,:,:,q0] = R
"""