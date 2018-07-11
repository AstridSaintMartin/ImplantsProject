#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from ggplot import *
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_mldata
import pickle
from ImageDataset import *
import seaborn as sns

#Dataset and dataloader
Trans=transforms.Compose([transforms.ToTensor()])
testset=KneeDataset(csv_file="DATA/Test.csv",dict_file="DATA/dictImages64x64.pkl", transform=Trans)
testloader= DataLoader(testset, batch_size=10000,shuffle=True)
rndperm = np.random.permutation(len(testset))


#
encoder=pickle.load(open("150encoderData","r"))
decoder=pickle.load(open("150decoderDATA","r"))
noise = torch.rand(1,64,64)


#Plot the graph
plt.gray()
fig = plt.figure( figsize=(20,7) )
figure=plt.figure( figsize=(20,7) )
for i in range(0,10):
    img,label=testset[rndperm[i]]
    print img.size()
    ax = fig.add_subplot(2,5,i+1, title='Digit: ' + str(label))
    image=img.numpy()
    ax.matshow(image[0,:,:], aspect="auto")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    img= Variable(torch.mul(img+0.25, 0.5 * noise))
    img=img.view(-1,64*64)
    decoded=decoder(encoder(img))
    decoded=decoded[0].resize(1,64,64)
    decoded=decoded.data.numpy()
    bx=figure.add_subplot(2,5,i+1, title='Hash: ' + str(label))
    bx.matshow(decoded[0], aspect="auto")
    bx.set_yticklabels([])
    bx.set_xticklabels([])
fig.savefig("100PanelKneeImage")
figure.savefig("100PanelReconstructed")
plt.close()
print "finish"

#latentspace vizualisation
xtest,label=next(iter(testloader))
xtest=Variable(xtest)
#img= Variable(torch.mul(img+0.25, 0.5 * noise))
xtest=xtest.view(-1,64*64)
encoded=encoder(xtest)
encoded=encoded.data.numpy()
feat_cols = [ 'pixel'+str(i) for i in range(encoded.shape[1]) ]
df = pd.DataFrame(encoded,columns=feat_cols)
df['Label'] = label

#Apply PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['PCA first component'] = pca_result[:,0]
df['PCA second component'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)

sns.palplot(sns.color_palette("cubehelix", 32))
g=sns.lmplot('PCA first component', 'PCA second component', data=df.loc[rndperm[:5000],:],hue="Label",fit_reg=False)
g.fig.set_size_inches(12,12)
# resize figure box to -> put the legend out of the figure
box = g.ax.get_position() # get position of figure
g.ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
plt.savefig("StackPCA")

#APPLY KPCA
kpca=KernelPCA(n_components=2, kernel="rbf", gamma=5)
kpca_result = kpca.fit_transform(df[feat_cols].values)
df['KPCA first component'] = pca_result[:,0]
df['KPCA second component'] = pca_result[:,1]
sns.palplot(sns.color_palette("cubehelix", 32))
g=sns.lmplot('KPCA first component', 'KPCA second component', data=df.loc[rndperm[:5000],:],hue="Label",fit_reg=False)
g.fig.set_size_inches(12,12)
# resize figure box to -> put the legend out of the figure
box = g.ax.get_position() # get position of figure
g.ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
plt.savefig("StackKPCA")


#Apply tsne
n_sne = 7000
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['t-SNE first component'] = tsne_results[:,0]
df_tsne['t-SNE second component'] = tsne_results[:,1]

#sns.palplot(sns.color_palette("cubehelix", 32))
g=sns.lmplot('t-SNE first component', 't-SNE second component', data=df_tsne.loc[rndperm[:5000],:],hue="Label",fit_reg=False)
g.fig.set_size_inches(12,12)
# resize figure box to -> put the legend out of the figure
box = g.ax.get_position() # get position of figure
g.ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
plt.savefig("StackTSNE")
