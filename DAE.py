#!/usr/bin/env python

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torch.autograd import Variable

class Autoencoder(nn.Module):
    def __init__(self, nz=2):
        super(Autoencoder, self).__init__()
        self.nz=nz
        self.encoder=nn.Sequential(
        nn.Linear(28*28,128),
        nn.Tanh(),
        nn.Linear(128,64),
        nn.Tanh(),
        nn.Linear(64,12),
        nn.Tanh(),
        nn.Linear(12,nz),
        )
        self.decoder=nn.Sequential(
        nn.Linear(nz,12),
        nn.Tanh(),
        nn.Linear(12,64),
        nn.Tanh(),
        nn.Linear(64,128),
        nn.Tanh(),
        nn.Linear(128,28*28),
        nn.Sigmoid()
        )
        
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return encoded, decoded
        
    def lossfunc(self, decoded,original):
        loss=nn.BCELoss()
        return loss(decoded, original)
