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
from DAAE_LINEAR import *
import pickle
import time


#auto-encoder parameters

batch_size=100
tiny=1e-15
max_epoch=40
latent_code_size=32
img_size=784
img_width=28
img_height=28
noise = torch.rand(batch_size,1,28,28)

# Set learning rates
global gen_lr
gen_lr= 0.0001
reg_lr = 0.00005


def train (autoencoder, train_loader):
    optimizer=torch.optim.Adam(autoencoder.parameters(),lr=0.005)
    for epoch in range(max_epoch):
        for step, (image, label) in enumerate(train_loader):
            image= torch.mul(image+0.25, 0.5 * noise)
            image=Variable(image.view(-1,28*28))
            label=Variable(label)
            encoded, decoded=autoencoder.forward(image)
            loss=autoencoder.lossfunc(decoded,image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
                
def train_DAAE(encoder,decoder,discriminator, train_loader):
    #encoder and decoder optimisers
    Q_optim_encoder=torch.optim.Adam(encoder.parameters(),lr=gen_lr,betas=(0.5, 0.999))
    P_optim=torch.optim.Adam(decoder.parameters(),lr=gen_lr,betas=(0.5, 0.999))
    #encoder and discriminator optimisers
    Q_optim_generator=torch.optim.Adam(encoder.parameters(),lr=reg_lr,betas=(0.5, 0.999))
    D_optim=torch.optim.Adam(discriminator.parameters(),lr=reg_lr,betas=(0.5, 0.999))
    #
    #
    BCEloss=[]
    Discriminatorloss=[]
    for epoch in range(max_epoch):
        if epoch==10:
            global gen_lr
            gen_lr=gen_lr/2
        for step, (image, label) in enumerate(train_loader):
            imageNoisy= torch.mul(image+0.25, 0.5 * noise)
            imageNoisy=Variable(imageNoisy.view(-1,28*28))
            label=Variable(label)
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()
            z_sample=encoder(imageNoisy)
            X_sample=decoder(z_sample)
            criterion=nn.BCELoss()
            recon_loss=criterion(X_sample.resize(100,1,28,28)+tiny, Variable(image).resize(100,1,28,28)+tiny)
            recon_loss.backward()
            Q_optim_encoder.step()
            P_optim.step()
            #
            #discriminator
            encoder.eval()
            Z_real=Variable(torch.randn(image.size()[0],32)*5.)
            D_real=discriminator(Z_real)
            Z_fake=encoder(imageNoisy)
            D_fake=discriminator(Z_fake)
            D_loss = -torch.mean(torch.log(D_real + tiny) + torch.log(1 - D_fake + tiny))
            D_loss.backward()
            D_optim.step()
            #
            #generator
            encoder.train()
            Z_fake=encoder(imageNoisy)
            D_fake=discriminator(Z_fake)
            G_loss= -torch.mean(torch.log(D_fake+ tiny))
            G_loss.backward()
            Q_optim_generator.step()        
            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % recon_loss.data[0],'| discriminator loss: %.4f'% D_loss.data[0])
                print "parameter learning", gen_lr
        BCEloss.append(recon_loss.data[0])
        Discriminatorloss.append(D_loss.data[0])
    fig=plt.figure()
    plt.plot(range(max_epoch),Discriminatorloss,label="discriminator loss")
    plt.plot(range(max_epoch),BCEloss,label="BCE reconstruction loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("losses")
    plt.gca().legend(('discriminator loss','BCE reconstruction loss'))
    plt.title("losses for 40 epochs with latent size of 32, DAAE linear and BCE loss")
    fig.savefig("losses for 40 epochs with latent size of 32, DAAE linear and BCE loss")






if __name__ == '__main__':

    start=time.time()
    #
    #MNIST dataset
    train_data=dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    test_data=dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    #Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

    encoder=Q_net(28*28,1000,32)
    decoder=P_net(28*28,1000,32)
    discriminator=D_net(500,32)
    #encoder=Q_net()
    #decoder=P_net()
    #discriminator=D_net(500,latent_code_size)

    print("begin training")
    train_DAAE(encoder,decoder,discriminator,train_loader)
    
    pickle.dump(encoder,open("32_encoder_DAAELINEAR_BCE_40","wb"))
    pickle.dump(decoder,open("32_decoder_DAAELINEAR_BCE_40","wb"))
    
    loaded_encoder=pickle.load(open("32_encoder_DAAELINEAR_BCE_40","r"))
    loaded_decoder=pickle.load(open("32_decoder_DAAELINEAR_BCE_40","r"))
    
    loaded_encoder.eval()
    loaded_decoder.eval()
    image=test_loader.dataset[10][0]
    image= Variable(torch.mul(image+0.25, 0.5 * noise))
    image=image.view(-1,28*28)
    #encoded, decoded=autoencoder.forward(image)
    decoded=loaded_decoder(loaded_encoder(image))
    
    figure=plt.figure()
    image=image[1].resize(1,28,28)
    image=image.data.numpy()
    plt.imshow(image[0],cmap="gray")
    figure.savefig("32_original_DAAELINEAR_BCE_40")
    plt.close()
    
    figure=plt.figure()
    decoded=decoded[1].resize(1,28,28)
    decoded=decoded.data.numpy()
    plt.imshow(decoded[0],cmap="gray")
    figure.savefig("32_decoded_DAAELINEAR_BCE_40")
    plt.close()
    
    end=time.time()
    print("execution time",end-start)

