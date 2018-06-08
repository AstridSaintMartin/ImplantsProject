#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from DAAE import Autoencoder

#auto-encoder parameters

batch_size=100
max_epoch=15
latent_code_size=2
img_size=784
img_width=28
img_height=28

def train (autoencoder, train_loader):
    optimizer=torch.optim.Adam(autoencoder.parameters(),lr=0.005)
    for epoch in range(max_epoch):
        for image, label in train_loader:
            image=Variable(image.view(-1,28*28))
            label=Variable(label)
            encoded, decoded=autoencoder.forward(image)
            loss=autoencoder.lossfunc(decoded,image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])


if __name__ == '__main__':
    #MNIST dataset

    train_data=dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    test_data=dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

#Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

    autoencoder=Autoencoder(latent_code_size)
    print("begin training")
    train(autoencoder,train_loader)
    
    

