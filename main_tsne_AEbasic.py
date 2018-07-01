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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

def visualize(autoencoder, test_loader):
    # get your latent space in 2D (should try t-SNE representation later on)
    images=test_data.test_data[:100]
    labels=test_data.test_labels[:100].numpy()
    images = Variable(images.view(-1, 28*28).type(torch.FloatTensor)/255.)
    encEx= autoencoder.forward(images)
    encEx=images
    encEx = PCA(n_components=12).fit_transform(encEx.data.numpy())
    encEx = TSNE(n_components=2, perplexity=5, verbose=2).fit_transform(encEx)
    plt.figure(figsize=(6, 6))
    #plt.scatter(encEx[:,0].data.numpy(),encEx[:,1].data.numpy(), c=labels)
    plt.scatter(encEx[:,0],encEx[:,1], c=labels)
    plt.colorbar()
    plt.savefig("regularisation5_rawdata_TSNE")
    plt.close()

if __name__ == '__main__':

    start=time.time()
    #
    #MNIST dataset
    train_data=dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    test_data=dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    #Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

    #autoencoder = Autoencoder(16)
    #train(autoencoder,train_loader)
    #pickle.dump(autoencoder,open("autoencoder,per=10,echo=40","wb"))
    autoencoder=pickle.load(open("32_encoder_DAAELINEAR_BCE_40","r"))
    autoencoder.eval()
    visualize(autoencoder,test_loader)
    
    print "finish"
    
