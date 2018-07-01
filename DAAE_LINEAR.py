#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms


#Encoder

class Q_net(nn.Module):
    def __init__(self,x_dim,N,z_dim):
        super(Q_net,self).__init__()
        self.lin1=nn.Linear(x_dim,N)
        self.lin2=nn.Linear(N,N)
        self.lin3=nn.Linear(N,z_dim)

    def forward(self,x):
        x=F.dropout(self.lin1(x),p=0.25, training=self.training)
        x=F.relu(x)
        x=F.dropout(self.lin2(x),p=0.25, training=self.training)
        x=F.relu(x)
        x=self.lin3(x)
        return x


#Decoder

class P_net(nn.Module):
    def __init__(self,x_dim,N,z_dim):
        super(P_net,self).__init__()
        self.lin1=nn.Linear(z_dim,N)
        self.lin2=nn.Linear(N,N)
        self.lin3=nn.Linear(N,x_dim)

    def forward(self,x):
        x=F.dropout(self.lin1(x),p=0.25,training=self.training)
        x=F.relu(x)
        x=F.dropout(self.lin2(x),p=0.25, training=self.training)
        x=F.relu(x)
        x=self.lin3(x)
        return F.sigmoid(x)



#Discriminator
class D_net(nn.Module):
    def __init__(self,N,z_dim):
        super(D_net,self).__init__()
        self.lin1=nn.Linear(z_dim,N)
        self.lin2=nn.Linear(N,N)
        self.lin3=nn.Linear(N,1)

    def forward(self,x):
        x=F.dropout(self.lin1(x),p=0.2,training=self.training)
        x=F.relu(x)
        x=F.dropout(self.lin2(x),p=0.2, training=self.training)
        x=F.relu(x)
        x=self.lin3(x)
        return F.sigmoid(x)






