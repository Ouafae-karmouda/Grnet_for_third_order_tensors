# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:58:58 2021

@author: adminlocal
"""


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from layers_fcts import call_reorthmap, call_ProjMap, call_ProjPooling, call_orthmap, call_FrMap, update_params_filters
from tlinalg import t_product_tensors, t_transpose

tenType = torch.cuda.FloatTensor 


class Grnet_tensors(torch.nn.Module):
 
    def __init__(self, num_classes, num_filters, d1, d0, kernel_size, strides):
        """
        d0 :  first dimension of input images
        filters : w_k of size (d1, d0)
        
        """
        super(Grnet_tensors, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_weights = []
        self.d0 = d0
        self.d1 = d1
        self.kernel_size = kernel_size
        self.strides = strides
        #self.fc_w = torch.nn.Parameter(Variable(torch.randn(rn_value, self.num_classes, device="cuda")))
        self.fc_b = torch.nn.Parameter(Variable(torch.randn(1, self.num_classes, device="cuda", requires_grad = True)))
        #initialisze weight_filters wuth random values from a random distribution with mean 0 and variance 1
    
        for i in range(self.num_filters):
            self.filter_weights.append(torch.nn.Parameter(Variable(torch.randn(self.d1, d0, device="cuda", 
                                                                              requires_grad = True))))
        self.params_filter_weights = torch.nn.ParameterList([ self.filter_weights[i] for i in range (self.num_filters)])
        
                    
    def forward(self, inputs):
        """
        inputs should be tensors
        """
        inputs = inputs.type(torch.FloatTensor).cuda()
        batch_size = inputs.shape[0]
        W1_c = [self.filter_weights[i].contiguous() for i in range(self.num_filters)]
        W1 = [W1_c[i].view(1, W1_c[i].shape[0], W1_c[i].shape[1]) for i in range(self.num_filters)]
        X1= [torch.matmul(W1[i], inputs) for i in range(self.num_filters)]
        X1_tensor = torch.stack(X1)
        X1_tensor = torch.stack(X1).view((X1_tensor.shape[1],
                                    X1_tensor.shape[2],
                                    X1_tensor.shape[0],
                                   X1_tensor.shape[3]))
        #X1_tensor = call_FrMap(inputs, X1)
        #print("X1_tensor.grad_fn",X1_tensor.grad_fn)
        #print("X1_tenosr.shape", X1_tensor.shape)
        X2 = call_reorthmap(X1_tensor)
        #print("X2.shape", X2.shape)

        #print(X2.device)
        #print("X2_tensor.grad_fn",X2.grad_fn)
        #X3 = call_ProjMap(X2)
        outputs = []
        for i in range(batch_size):
            outputs.append(t_product_tensors(X2[i,:,:,:], t_transpose(X2[i,:,:,:]) ))
        X3 = torch.stack(outputs).cuda()
        #print("X3.shape", X3.shape)
        #print("X3_tensor.grad_fn",X3.grad_fn)
        #X4 = call_ProjPooling(X3, self.kernel_size , self.strides)
        m = torch.nn.AvgPool1d(self.kernel_size, self.strides)
        outputs =  []
        for i in range(batch_size):
            out_pool = m(X3[i])
            outputs.append(out_pool)
        X4 = torch.stack(outputs)
        #print("X4.shape", X4.shape)
        #print("X4_tensor.grad_fn",X4.grad_fn)

        #[batch_size, d1, d2, q_prime] = X4.shape
        #m1 = X4.shape[2] - 2
        X5 = call_orthmap(X4)
        #print("X5.shape", X5.shape)
        #print("X5_tensor.grad_fn",X5.grad_fn)

        #X6 = call_ProjMap(X5)
        outputs = []
        for i in range(batch_size):
            outputs.append((t_product_tensors(X5[i,:,:,:], t_transpose(X5[i,:,:,:]) )))
        X6 = torch.stack(outputs).cuda()
        #print("X6_tensor.grad_fn",X6.grad_fn)
        #print("X6.shape", X6.shape)
        FC = X6.view([batch_size, -1])
        #print('FC_grad', FC.grad.shape)
        #print("FC.shape", FC.shape)
        first_dim_dense_layer = FC.shape[-1]
        #self.params.append(torch.nn.Parameter(Variable(torch.randn(first_dim_dense_layer, self.num_classes, device="cuda",requires_grad = True))))
        #self.params[0]=self.fc_w
        self.fc_w = torch.nn.Parameter(Variable(torch.randn(first_dim_dense_layer, self.num_classes, device="cuda",
                                                            requires_grad = True)))

        logits = torch.add(torch.matmul(FC.float(), self.fc_w.float()), self.fc_b.float())
        output = F.log_softmax(logits, dim=-1)
        
        return output
                    
    def update_param(self, lr):

            new_W1 = []
            for i in range(self.num_filters):
                #new_W1.append(utils.update_params_model(W1_np[i], eugrad_W1[i],lr))
                new_W1.append(update_params_filters(self.filter_weights[i].data, 
                                                    self.filter_weights[i].grad.data,lr))
                self.filter_weights[i].data.copy_(tenType(new_W1[i]))
            
            with torch.no_grad():
                self.fc_w.data -= lr * self.fc_w.grad.data
                self.fc_b -= lr*self.fc_b.grad.data   
                
            #set gradients to zero manually
            for i in range(self.num_filters):
                self.filter_weights[i].grad.data.zero_()
            self.fc_w.grad.data.zero_()
            self.fc_b.grad.data.zero_()
        
            
        
    def training_step(self, batch):
        images, labels = batch 
        out = self.forward(images)          # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss, out
    

    
    def validation_step(self, batch):
        images, labels = batch 
        out = self.forward(images)                    # Generate predictions
        print("out shape",out.shape)
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    

    
def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds==labels)/len(preds))
    
    
