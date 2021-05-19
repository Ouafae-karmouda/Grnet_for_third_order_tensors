# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:53:06 2021

@author: adminlocal
"""
import torch
torch.cuda.empty_cache()
from model_grnet_tensor import Grnet_tensors
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
from torch.utils.data import Subset
import torchvision

num_classes = 10
num_filters = 16
#d1 to choose > q 
d1 = 30
q =28
d0 = 28
m0 = 16
#d0: first dimension of data





class MNist_grass(Dataset):
    
    def __init__(self):
        #data loading
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        y_train = torch.tensor(y_train, dtype=torch.long, device="cuda")
        X_grassmann_train = np.load('./data/X_grassmann_train.npy')
        #X_grassmann_test = np.load('./data/X_grassmann_test.npy')
        #∟reshape images into (28,28)
        X_grassmann_train = X_grassmann_train.reshape(-1, 28, 28)
        #X_grassmann_test = X_grassmann_test.reshape(-1, 28, 28)
        self.x = torch.from_numpy(X_grassmann_train).cuda()
        self.y = y_train
        self.n_samples = X_grassmann_train.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
   
dataset = MNist_grass()

#create a training and a validation set
train_ds, val_ds = random_split(dataset, [50000, 10000])

#create the loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle = True)
val_loader =  DataLoader(val_ds, batch_size=128)


#X_grassmann_train, y_train = shuffle(X_grassmann_train, y_train)
#X_grassmann_test, y_test = shuffle(X_grassmann_test, y_test)



first_dim_dense_layer = d1 *m0 * q
model = Grnet_tensors(num_classes, num_filters, d1, d0, first_dim_dense_layer).cuda()
#inputs = X_train_tensors[:50]
#outputs = model.forward(inputs)

###Train the model
loss_fn = F.cross_entropy



def evaluate(model, val_loader):
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds==labels)/len(preds))

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history
"""
for batch in train_loader:
  images, labels = batch
  outputs = model.forward(images)
  print("outputs :\n", outputs)
  probs = F.softmax(outputs)
  print(probs)
  max, preds = torch.max(probs, dim=1)
  acc = accuracy(probs, labels)
  break
"""  

history1 = fit(5, 0.001, model, train_loader, val_loader)
#result0 = evaluate(model, val_loader)
#result0