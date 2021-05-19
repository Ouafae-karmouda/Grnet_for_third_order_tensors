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

class FrMap(Function):
    """
    @staticmethod
    def forward(ctx, input, model):
        
        #: batch of tensors of size (d0, q)
       
        print("Forward FrMap")
        
        W1_c = [model.filter_weights[i].contiguous() for i in range(model.num_filters)]
        W1 = [W1_c[i].view(1, W1_c[i].shape[0], W1_c[i].shape[1]) for i in range(model.num_filters)]
        X1_list= [torch.matmul(W1[i], input) for i in range(model.num_filters)]
        print(X1_list[0].grad_fn)
        X1_tensor = torch.stack(X1_list)
        
        #X1_tensor = X1_tensor.view((X1_tensor.shape[1],
        #                            X1_tensor.shape[2],
        #                            X1_tensor.shape[0],
        #                            X1_tensor.shape[3]))
        ctx.save_for_backward(input)
        return X1_tensor
    """
    @staticmethod
    def forward(ctx, input, X1_list):
        """
        inputs: batch of tensors of size (d0, q)
        """ 
        print("Forward FrMap")
        X1_tensor = torch.stack(X1_list)
        ctx.save_for_backward(input)
        return X1_tensor
    
   
    def backward(ctx, grad_outputs):
        
        #grad_outputs is of size (d1, m0,q)
       
        print("Backward FrMap")
        input= ctx.saved_tensors
        [batch_size, d1, m0,q] = grad_outputs.shape
        grads = torch.zeros_like(grad_outputs)
        
        for m in range(m0):
            
            grads[:,:,m,:] = torch.matmul(grad_outputs[:,:,m,:], input.transpose(1,0))
            #ctx.model.filter_weights[m].grad =  grads[:,:,m,:].copy()
            ctx.model.filter_weights[m].grad.fill_(grads[:,:,m,:])
            print("grads")
        return grads
   
    
            
            
            
class Reorth(Function):
        
        @staticmethod
        def forward(ctx, inputs):
            """
            inputs: batch of tensors X1 from the FRMap layer of size (d1, m0, q)
            m0 : number of filters
            d1 > q : to have a reduced form of QR
            """
            [batch_size, d1, m0, q] = inputs.shape
            ctx.Q_fft = torch.zeros_like(inputs).type(tenType).cuda()
            ctx.R_fft =  torch.zeros(batch_size, q, m0, q).cuda()
            ctx.m0 = m0
            [batch_size, d1, m0, q] = inputs.shape
            #Q_fft = torch.zeros_like(inputs).type(tenType).cuda()
           # R_fft =  torch.zeros(batch_size, q, m0, q).cuda()
            ctx.m0 = m0
            
            print("Forward Reorth")
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
            ctx.Q_fft = Q_tensor
            ctx.R_fft = R_tensor
            Q_ifft_tensor = np.fft.ifft(Q_tensor.cpu()).real
            return  torch.from_numpy(Q_ifft_tensor).cuda()
        
        @staticmethod
        def backward(ctx, grad_outputs):
            """
            grad_outputs : gradients of the next layer which is the ProjMap of size (batch_size, d1, d1, q)
    
            """
            print("Backward Reorth")
            print(grad_outputs.shape)
            [batch_size, d1, d1, q] = grad_outputs.shape
            print("d1 reorth",d1)
            d1 = 28
            dLdQ = grad_outputs.type(tenType).cuda()
            grad = torch.zeros_like(grad_outputs).type(tenType).cuda() #grad and gradout same dims
        
        
            for m in range(m0): 
                #print("m", m)
                #print("Q_fft shape", ctx.Q_fft.shape)
                Q = ctx.Q_fft[:,:,m,:]
                R = ctx.R_fft[:,:,m,:]
                R_inv = torch.pinverse(R)
                #print("Q.shape", Q.shape)
                #print("eye(d1.shape", torch.eye(d1).cuda().shape )
                list_eye = [torch.eye(d1).cuda() for i in range(batch_size)]
                tensors_eye = torch.stack(list_eye)
                S = tensors_eye - torch.matmul(Q, Q.transpose(1,2) )
                #print("S.shape", S.shape)
                ele_1 = torch.matmul(S.transpose(1,2), dLdQ[:, :, m, :])
                temp = torch.matmul(Q.transpose(1,2), dLdQ[:, :, m, :])
                temp_bsym = torch.tril(temp) - torch.tril(temp.transpose(1, 2))
                ele_2 = torch.matmul(Q, temp_bsym)
                grad[:, :,m, :] = torch.matmul(ele_1+ele_2, R_inv.transpose(1,2))
            
            grad_ifft = np.fft.ifft(grad.cpu()).real
           
            return torch.from_numpy(grad_ifft).cuda()
        

class OrthMap(Function):

    @staticmethod
    def forward(ctx, input):
        """
        inputs batch of tensors from the ProjPooling of size (d2, d2, q)
        m1 : the nbr of matrices to keep from the U matrix of t-SVD 
        """
        m1 =10
        print("Forward OrthMap")
        [batch_size, d2, d2, q] = input.shape
        ctx.d2 = d2
        ctx.q = q
        ctx.m1 = m1
        ctx.U_fft_tensor =  torch.zeros(batch_size, d2,d2,q).type(tenType).cuda()
        ctx.S_fft_tensor = torch.zeros(batch_size, d2).type(tenType).cuda()
        #[batch_size, d2, d2, q] = inputs.shape
        #self.q = q
        #self.m1 = m1
        #self.d2 = d2
        inputs_fft = np.fft.fft(input.cpu().detach().numpy())
        inputs_fft = torch.from_numpy(inputs_fft).cuda()
        #U_tensor =  torch.zeros(batch_size, d2,m1,q).type(tenType).cuda()
        list_S = []
        for i in range(q):
            (U, S, V) = torch.svd(inputs_fft[:,:, :, i], some=True)
            #self.U_fft_tensor[:,:,:,i] = U[:,:,:m1]
            ctx.U_fft_tensor[:,:,:,i] = U
            list_S.append(S)
        ctx.S_fft_tensor = torch.stack(list_S)
        U_ifft_tensor = np.fft.ifft(ctx.U_fft_tensor.cpu()).real
            
        return  torch.from_numpy(U_ifft_tensor[:,:,:m1,:]).cuda()
    
    
    @staticmethod
    def backward(ctx, grad_outputs):
        """
        grad_outputs is of shape (batch_size,d2, d2,q)
        """
        print("Backward OrthMap")
        print(grad_outputs.shape)
        dLdU = grad_outputs.type(tenType).cuda() #needs concatenation of zeros to complete [bs, d1, d1]
        grad = torch.zeros((dLdU.shape[0], dLdU.shape[1], ctx.d2, ctx.q)).type(tenType).cuda()
        batch_size = grad_outputs.shape[0]
        d2 = ctx.S_fft_tensor.shape[-1]
        q =  ctx.S_fft_tensor.shape[0]
        for i in range(ctx.q):
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
                 temp = torch.cat((temp, torch.zeros((batch_size, d2, d2-dLdU[:,:,:,i].shape[-1]), dtype=torch.float32).cuda()), dim=2)
                 #print("temp.shape", temp.shape)
                 #print("q", q)
                 #print("d2", d2)
                 #temp = torch.cat((temp, torch.zeros((self.d2, self.m1), dtype=torch.float32)), dim=1)
                 #print("temps.shape", temp.shape)
                 #print("dLdU.shape", dLdU.shape)
                 #print("temp", temp.device)
                 K = K.transpose(1,0)
                 #print("K.shape", K.shape)
                 temp = K*temp
                 #print("ctx.U_fft_tensor[:,:,:,i]", ctx.U_fft_tensor[:,:,:,i].dtype)
                 #print("temp", temp.dtype)
                 temp = temp.type('torch.cuda.FloatTensor')
                 temp = torch.matmul(ctx.U_fft_tensor[:,:,:,i], temp)
                 
                 temp = torch.matmul(temp, U_fft_tensor_t[:,:,:,i] )
                 grad[:, :, :,i] = temp
                 
        grad_ifft = np.fft.ifft(grad.cpu()).real
        return torch.from_numpy(grad_ifft).cuda()
         

            
        
    
    
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
     
def call_FrMap(inputs, model):
    frmap= FrMap.apply
    return  frmap(inputs,model)
        
def call_reorthmap(inputs):
    reorth = Reorth(inputs)
    return reorth.apply(inputs)

def call_orthmap(inputs):
    orth = OrthMap(inputs)
    return orth.apply(inputs)
    
def reim_grad_fct(W, euc_grad):
    """
    reim_grad = euc_grad - euc_grad * W^T * W
    """
    print("W", W.shape)
    W_t = W.transpose(1, 0)
    temp = torch.matmul(W_t, W)
    reim_grad = euc_grad - torch.matmul(euc_grad, temp)
    #reim_grad = retraction(W - lamda* grad_tilde)
    
    return reim_grad
    
def update_params_filters(W, euc_grad,lr):
    """
    W_up: update of W_up = retraction(W - lr*reim_grad)
    """
    print('compute reim gradients')
    reim_grad = reim_grad_fct(W, euc_grad)
    #W_up = retraction1(W - lr * reim_grad)
    W_up = W - lr * reim_grad 
    return W_up


def retraction1(X):
    '''
    Projecto onto the manifold of fixed rank matrices
    Projection = sum of ui.vi.T for rank r first terms
    '''
    X_np = X.cpu().numpy()
    u, d, v = np.linalg.svd(X_np)
    rank = np.linalg.matrix_rank(X_np)
    # to chkck d[0] or abs(d[0])
    x = d[0]*np.matmul(np.expand_dims(u[:, 0], axis = 1), np.expand_dims(v[:, 0].transpose(), axis=0))
    for i in range(1, rank):
        x += d[i]*np.matmul(np.expand_dims(u[:, i], axis = 1), np.expand_dims(v[:, i].transpose(), axis=0))
    
    return torch.from_numpy(x).cuda()


def update_params_filters1(lr, W, euc_grad):
    """
    W_up: update of W_up = retraction(W - lr*reim_grad)
    """
    
    #reim_grad = reim_grad_fct(W, euc_grad)
    W_up = W - lr * euc_grad
    
    return W_up


def retraction(M):
    
    """
    projection of M onto the psd manifold
    
    """
    print("M.shape", M.shape)
    eig_val, eig_vectors = np.linalg.eig(M.cpu().detach().numpy())
    abs_eig_val = abs(eig_val)
    
    eig_val = torch.from_numpy(eig_val)
    eig_vectors = torch.from_numpy(eig_vectors)
    proj_M = torch.matmul(eig_vectors, torch.matmul(torch.from_numpy((np.diag(abs_eig_val))), eig_vectors.transpose(1,0)))
    
    return proj_M.cuda()


"""
 #tronquer à la dimension de filtrage
            
            for q0 in range(q):
                (Q, R) = torch.qr(inputs_fft[:,:, :, q0], some=True)
                Q_tensor[:,:,:,q0] = Q
                R_tensor[:,:,:,q0] = R
"""