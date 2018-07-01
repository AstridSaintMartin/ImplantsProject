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
import pickle
import time



class MNISTencoder(nn.Module):
    def __init__(self):
        super(MNISTencoder, self).__init__()
        self._name = 'mnistE'
        self.shape = (1, 28, 28)
        self.dim = 32
        convblock = nn.Sequential(
                nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                )
        self.main = convblock
        self.output = nn.Linear(4*4*4*self.dim, self.dim)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        return out.view(-1, self.dim)

class MNISTdiscriminator(nn.Module):
    def __init__(self):
        super(MNISTdiscriminator, self).__init__()
        self._name = 'mnistD'
        self.shape = (1, 28, 28)
        self.dim = 32
        convblock = nn.Sequential(
                nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                )
        self.main = convblock
        self.output = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        return out.view(-1)


class MNISTgenerator(nn.Module):
    def __init__(self):
        super(MNISTgenerator, self).__init__()
        self._name = 'mnistG'
        self.dim = 32
        self.in_shape = int(np.sqrt(self.dim))
        self.shape = (self.in_shape, self.in_shape, 1)
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 4*4*4*self.dim),
                nn.ReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(4*self.dim, 2*self.dim, 5),
                nn.ReLU(True),
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(2*self.dim, self.dim, 5),
                nn.ReLU(True),
                )
        deconv_out = nn.ConvTranspose2d(self.dim, 1, 8, stride=2)
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        #output = F.dropout(output, p=0.3, training=self.training)
        output = output.view(-1, 4*self.dim, 4, 4)
        output = self.block1(output)
        #output = F.dropout(output, p=0.3, training=self.training)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        #output = F.dropout(output, p=0.3, training=self.training)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, 784)
        
        
#auto-encoder parameters

batch_size=100
tiny=1e-15
max_epoch=20
latent_code_size=32
img_size=784
img_width=28
img_height=28
noise = torch.rand(batch_size,1,28,28)

def train(netG,netE,netD):
    optimizerD =torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9))
    optimizerG =torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9))
    optimizerE =torch.optim.Adam(netE.parameters(), lr=2e-4, betas=(0.5, 0.9))  
    ae_criterion = nn.BCELoss()
    for epoch in range(max_epoch):
        for step, (image, label) in enumerate(train_loader):
            """ Update AutoEncoder """
            netG.zero_grad()
            netE.zero_grad()
            imageNoisy= torch.mul(image+0.25, 0.5 * noise)
            imageNoisy=Variable(imageNoisy)
            encoding = netE(imageNoisy)
            fake = netG(encoding)
            ae_loss = ae_criterion(fake,Variable(image))
            ae_loss.backward()
            optimizerE.step()
            optimizerG.step()
            #""" Update D network """
            Z_real=Variable(torch.randn(image.size()[0],4*4*4)*5.)
            D_real=netD(Z_real)
            Z_fake=netE(imageNoisy)
            D_fake=netD(Z_fake)
            D_loss = -torch.mean(torch.log(D_real + tiny) + torch.log(1 - D_fake + tiny))
            D_loss.backward()
            optimizerD.step()
           # Update generator network (GAN)
            Z_fake=encoder(imageNoisy)
            D_fake=discriminator(Z_fake)
            G_loss= -torch.mean(torch.log(D_fake+ tiny))
            G_loss.backward()
            optimizerG.step()
            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % ae_loss.data[0],'| discriminator loss: %.4f'% D_loss.data[0])





if __name__ == '__main__':
    #MNIST dataset
    train_data=dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    test_data=dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    #Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)


    netG=MNISTgenerator()
    netE=MNISTencoder()
    netD=MNISTdiscriminator()
    train(netG,netE,netD)
    pickle.dump(netG,open("modelnetG","wb"))
    pickle.dump(netE,open("modelnetE","wb"))
    pickle.dump(netD,open("modelnetD","wb"))

