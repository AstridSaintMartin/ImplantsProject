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
from ImageDataset import *


class DAE(nn.Module):
    def __init__(self,latent_dim,imsize,number_filter,sigma):
        super(DAE,self).__init__()
        self.insize=imsize/2**4
        self.sigma=sigma
        
        #Encoder
        self.enc1=nn.Conv2d(1,number_filter,5,stride=2, padding=2)
        self.enc2=nn.Conv2d(number_filter,number_filter*2, 5, stride=2, padding=2)
        self.enc3=nn.Conv2d(number_filter*2,number_filter*4, 5, stride=2, padding=2)
        self.enc4=nn.Conv2d(number_filter* 4, number_filter* 8, 5, stride=2, padding=2)
        self.enc5=nn.Linear((number_filter*8)*self.insize*self.insize,latent_dim)
        
        
        #decoder
        self.dec1 = nn.Linear(latent_dim, (number_filter*8)*self.insize*self.insize)
        self.dec2 = nn.ConvTranspose2d(number_filter * 8, number_filter * 4, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(number_filter * 4, number_filter * 2, 3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ConvTranspose2d(number_filter * 2, number_filter, 3, stride=2, padding=1, output_padding=1)
        self.dec5 = nn.ConvTranspose2d(number_filter, 1, 3, stride=2, padding=1, output_padding=1)
        
    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = x.view(x.size(0), -1)
        x = self.enc5(x)
    
    def decode(self,z):
        z = F.relu(self.dec1(z))
        z = z.view(z.size(0), -1, self.inSize, self.inSize)
        z = F.relu(self.dec2(z))
        z = F.relu(self.dec3(z))
        z = F.relu(self.dec4(z))
        z = F.sigmoid(self.dec5(z))
        
    def corrupt(self, x):
        noise = self.sigma * Variable(torch.randn(x.size())).type_as(x)
        return x + noise
        
    def forward(self, x):
        x_corr = self.corrupt(x)
        z = self.encode(x_corr)
        b=self.decode(z)
        return z, b


dset=KneeDataset("DATA/DatasetKnee.csv","DATA/dictImages64x64.pkl")
dataloader = DataLoader(dset, batch_size=4,shuffle=True)
img,label=next(iter(dataloader))
autoencoder=DAE(20,64,16,0.1)
v=Variable(img,requires_grad=True)
print v.size()
z,x=autoencoder.forward(v)