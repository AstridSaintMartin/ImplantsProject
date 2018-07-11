#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
import numpy as np
from matplotlib import pyplot as plt
from DAAE_LINEAR import *
from ImageDataset import *
import pickle

#auto-encoder parameters

batch_size=97
tiny=1e-15
max_epoch=150
latent_code_size=32
img_size=64*64
img_width=64
img_height=64
noise = torch.rand(batch_size,1,64,64)

# Set learning rates
global gen_lr
gen_lr= 0.0001
reg_lr = 0.00005

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
    Gloss=[]
    for epoch in range(max_epoch):
        if epoch==10:
            global gen_lr
            gen_lr=gen_lr/2
        for step, (image, label) in enumerate(train_loader):
            imageNoisy= torch.mul(image+0.25, 0.5 * noise)
            imageNoisy=Variable(imageNoisy.view(-1,64*64))
            label=Variable(label)
            encoder.zero_grad()
            decoder.zero_grad()
            discriminator.zero_grad()
            z_sample=encoder(imageNoisy)
            X_sample=decoder(z_sample)
            criterion=nn.BCELoss()
            recon_loss=criterion(X_sample.resize(batch_size,1,64,64)+tiny, Variable(image).resize(batch_size,1,64,64)+tiny)
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
            if step % 10 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % recon_loss.data[0],'| discriminator loss: %.4f'% D_loss.data[0])
                print "parameter learning", gen_lr
        BCEloss.append(recon_loss.data[0])
        Discriminatorloss.append(D_loss.data[0])
        Gloss.append(G_loss.data[0])
    pickle.dump(BCEloss,open("BCEloss","wb"))
    pickle.dump(Discriminatorloss,open("Discriminatorloss","wb"))
    pickle.dump(Gloss,open("Gloss","wb"))


if __name__ == '__main__':


    #prepare data
    normalize = transforms.Normalize(mean=[20], std=[30])
    Trans=transforms.Compose([transforms.ToTensor()])
    dset=KneeDataset(csv_file="DATA/train.csv",dict_file="DATA/dictImages64x64.pkl", transform=Trans)
    dataloader = DataLoader(dset, batch_size=97,shuffle=True)
    testset=KneeDataset(csv_file="DATA/test.csv",dict_file="DATA/dictImages64x64.pkl", transform=Trans)
    testloader= DataLoader(testset, batch_size=97,shuffle=True)
    encoder=Q_net(64*64,1000,32)
    decoder=P_net(64*64,1000,32)
    discriminator=D_net(500,32)
    print("begin training")
    train_DAAE(encoder,decoder,discriminator,dataloader)
    pickle.dump(encoder,open("150encoderData","wb"))
    pickle.dump(decoder,open("150decoderDATA","wb"))
    encoder=pickle.load(open("150encoderData","r"))
    decoder=pickle.load(open("150decoderDATA","r"))
    img, label=next(iter(testloader))
    f=plt.figure()
    im=img.numpy()
    plt.imshow(im[80,0,:,:],cmap="gray")
    f.savefig("original")
    image= Variable(torch.mul(img+0.25, 0.5 * noise))
    image=image.view(-1,64*64)
    decoded=decoder(encoder(image))
    print decoded.size()
    decoded=decoded[80].resize(1,64,64)
    dec=decoded.data.numpy()
    fig=plt.figure()
    plt.imshow(dec[0], cmap="gray")
    fig.savefig("decoded")
