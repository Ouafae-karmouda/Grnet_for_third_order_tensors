# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:58:58 2021

@author: adminlocal
"""


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from layers_fcts import Reorth, call_reorthmap
import torch.nn.functional as F


class Grnet_tensors(torch.nn.Module):
 
    def __init__(self, num_classes, num_filters, d1, d0, first_dim_dense_layer):
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
        self.fc_w = torch.nn.Parameter(Variable(torch.randn(first_dim_dense_layer, self.num_classes, device="cuda")))
        self.fc_b = torch.nn.Parameter(Variable(torch.randn(1, self.num_classes, device="cuda")))
        #initialisze weight_filters wuth random values from a random distribution with mean 0 and variance 1 
        for i in range(self.num_filters):
            self.filter_weights.append(torch.nn.Parameter(Variable(torch.randn(self.d1, d0, device="cuda"),
                                                                   requires_grad = True)))
            #self.register_parameter(name = 'filter', param=self.filter_weights[i])
            
            self.register_parameter(name = 'filter [{}]'.format(i), param=self.filter_weights[i])
        #self.weight = torch.nn.Parameter( torch.FloatTensor(7, 32, 32)).to('cud        
        #self.params = torch.nn.ParameterList([ self.filter_weights[i] for i in range (self.num_filters)])
        
            
    def forward(self, inputs):
        """
        inputs should be tensors
        W1 : sortie de la FRMap layer
        """
        inputs = inputs.type(torch.FloatTensor).cuda()
        batch_size = inputs.shape[0]
        W1_c = [self.filter_weights[i].contiguous() for i in range(self.num_filters)]
        W1 = [W1_c[i].view(1, W1_c[i].shape[0], W1_c[i].shape[1]) for i in range(self.num_filters)]
        X1= [torch.matmul(W1[i], inputs) for i in range(self.num_filters)]
        X1_tensor = torch.stack(X1)
        
        X1_tensor = X1_tensor.view((X1_tensor.shape[1],
                                    X1_tensor.shape[2],
                                    X1_tensor.shape[0],
                                    X1_tensor.shape[3]))
        #print(X1_tensor.dtype)
        #reorth = Reorth(X1_tensor)
        #X2 = reorth.forward(X1_tensor)
        X2 = call_reorthmap(X1_tensor)
        FC = X2.view([batch_size, -1])
        print("FC.shape", FC.shape)
        logits = torch.add(torch.matmul(FC.float(), self.fc_w.float()), self.fc_b.float())
        output = F.log_softmax(logits, dim=-1)
        
        return output
    
    def training_step(self, batch):
        images, labels = batch 
        out = self.forward(images)          # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    

    
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