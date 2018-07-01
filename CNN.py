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



#Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net,self).__init__()
        self.enc_cnn_1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)
        self.enc_cnn_2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
    def forward(self,images):
        code = self.enc_cnn_1(images)
        m=nn.MaxPool2d(2, stride=2)
        code = m(F.relu(code))
        code = self.enc_cnn_2(code)
        code = F.relu(code)
        m=nn.MaxPool2d(2, stride=1)
        code=m(code)
        return code


class P_net(nn.Module):
    def __init__(self):
        super(P_net,self).__init__()
        self.dec_linear_1 = nn.ConvTranspose2d(8, 16, 3, stride=2)
        self.dec_linear_2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)
        self.conv3=nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)


    def forward(self,code):
        out = F.relu(self.dec_linear_1(code))
        out = F.relu(self.dec_linear_2(out))
        out = F.tanh(self.conv3(out))
        return out

#Discriminator
class D_net(nn.Module):
    def __init__(self,N,z_dim):
        super(D_net,self).__init__()
        self.lin1=nn.Linear(z_dim,N)
        self.lin2=nn.Linear(N,N)
        self.lin3=nn.Linear(N,1)

    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=F.dropout(self.lin1(x),p=0.2,training=self.training)
        x=F.relu(x)
        x=F.dropout(self.lin2(x),p=0.2, training=self.training)
        x=F.relu(x)
        x=self.lin3(x)
        return F.sigmoid(x)










def train_totalCNN(encoder,decoder,discriminator,train_loader):
    #
    #encoder and decoder optimisers
    Q_optim_encoder=torch.optim.Adam(encoder.parameters(),lr=gen_lr,betas=(0.9, 0.999))
    P_optim=torch.optim.Adam(decoder.parameters(),lr=gen_lr,betas=(0.9, 0.999))
    #
    #encoder and discriminator optimisers
    Q_optim_generator=torch.optim.Adam(encoder.parameters(),lr=reg_lr,betas=(0.9, 0.999))
    D_optim=torch.optim.Adam(discriminator.parameters(),lr=reg_lr,betas=(0.9, 0.999))
    #
    BCEloss=[]
    Discriminatorloss=[]
    for epoch in range(max_epoch):
        for image,label in train_loader:
            image_n = torch.mul(image+0.25, 0.5 * noise)
            image=Variable(image.view(image.size(0), -1))
            image_n=Variable(image_n.view(100,1,28,28))
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()
            z_sample=encoder(image_n)
            X_sample=decoder(z_sample)
            criterion=nn.BCELoss()
            recon_loss=criterion(X_sample+tiny, image+tiny)
            recon_loss.backward()
            Q_optim_encoder.step()
            P_optim.step()
            #discriminator
            encoder.eval()
            Z_real=Variable(torch.randn(image.size()[0],32)*5.)
            D_real=discriminator(Z_real)
            Z_fake=encoder(image_n)
            D_fake=discriminator(Z_fake)
            D_loss = -torch.mean(torch.log(D_real + tiny) + torch.log(1 - D_fake + tiny))
            D_loss.backward()
            D_optim.step()
            #generator
            encoder.train()
            Z_fake=encoder(image_n)
            D_fake=discriminator(Z_fake)
            G_loss= -torch.mean(torch.log(D_fake+ tiny))
            G_loss.backward()
            Q_optim_generator.step()
        print('epoch',i,'| reconstruction loss ', recon_loss.data[0],'| discriminator loss ', D_loss.data[0])
        BCEloss.append(recon_loss.data[0])
        Discriminatorloss.append(D_loss.data[0])
    figure=plt.figure()
    plt.plot(range(max_epoch),Discriminatorloss,label="discriminator loss")
    plt.plot(range(max_epoch),BCEloss,label="BCE reconstruction loss")
    plt.title("losses for 20 epochs with latent size of 32, total CNN and BCE loss")
    fig.savefig("losses for 20 epochs with latent size of 32, total CNN and BCE loss")
