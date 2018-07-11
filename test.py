#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torch.autograd import Variable
from torch import optim
from torch.nn.functional import binary_cross_entropy as bce
from torchvision import transforms, utils
import numpy as np
from ImageDataset import *
from encoderData import *
import pickle
import matplotlib.pyplot as plt

#Parameters
batchSize=80
maxEpoch=100
latent_dim=100
sigma=0.1
LR=1e-3
Mom=0.9



def train(autoencoder, discriminator, trainloader):
    #Create optimizers
    optimDAE = optim.Adam(autoencoder.parameters(), lr = LR)  
    optimDIS = optim.Adam(discriminator.parameters(), lr = LR)
    #Keeping track of training
    losses = {'enc': [], 'rec': [], 'dis':[], 'test rec':[]}
    for epoch in range(maxEpoch):
        epochEncLoss=0
        epochRecLoss=0
        epochDisLoss=0
        for i, (image, label) in enumerate(trainloader):
            image=Variable(image)
            zFake,xRec=autoencoder.forward(image)
            # clac losses
            recLoss = autoencoder.rec_loss(xRec, image, loss='MSE')  #loss='BCE' or 'MSE'
            encLoss = discriminator.gen_loss(zFake)
            disLoss = discriminator.dis_loss(zFake)
            daeLoss = recLoss + 1* encLoss
            #Do update
            optimDIS.zero_grad()
            disLoss.backward()
            optimDIS.step()
            optimDAE.zero_grad()
            daeLoss.backward()
            optimDAE.step()
            # storing losses for plotting later
            epochEncLoss+=encLoss.data[0]
            epochRecLoss+=recLoss.data[0]
            epochDisLoss+=disLoss.data[0]
            if (i+2)%10 == 0:
                print '[%d, %d] enc: %0.5f, rec: %0.5f, dis: %0.5f' % (epoch, i, encLoss.data[0], recLoss.data[0], disLoss.data[0])
        # storing losses for plotting later
        losses['enc'].append(epochEncLoss/i)
        losses['rec'].append(epochRecLoss/i)
        losses['dis'].append(epochDisLoss/i)
    pickle.dump(losses,open("100losses.pkl","wb"))

if __name__ == '__main__':
    #prepare data
    normalize = transforms.Normalize(mean=[20], std=[30])
    Trans=transforms.Compose([transforms.ToTensor()])
    dset=KneeDataset(csv_file="train.csv",dict_file="dictImages64x64.pkl", transform=Trans)
    dataloader = DataLoader(dset, batch_size=80,shuffle=True)
    #Create objects
    #autoencoder=DAE(latent_dim,64,10,sigma)
    #discriminator=Discriminator(latent_dim, autoencoder.norm_prior)
    #train(autoencoder, discriminator, dataloader)
    #pickle.dump(autoencoder,open("100BCEautoencoder.pkl","wb"))
    autoencoder=pickle.load(open("100BCEautoencoder.pkl","r"))
    img,label=next(iter(dataloader))
    print torch.mean(img)
    zfake,xRec=autoencoder.forward(Variable(img))
    img=img.numpy()
    xRec=xRec.data.numpy()
    figure=plt.figure()
    plt.imshow(img[0,0,:,:],cmap="gray")
    figure.savefig("100BCEoriData")

    fig=plt.figure()
    plt.imshow(xRec[0,0,:,:],cmap="gray")
    fig.savefig("100BCExRec")
    print "finish"
