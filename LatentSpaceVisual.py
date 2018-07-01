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
from ggplot import *
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_mldata
import pickle

#MNIST dataset
train_data=dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
test_data=dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
#
xtest=test_data.test_data.numpy()
ytest=test_data.test_labels.numpy()
xtest.resize(xtest.shape[0],xtest.shape[1]*xtest.shape[2])
xtest=Variable(torch.from_numpy(xtest))
#Take latent space
autoencoder=pickle.load(open("32_encoder_DAAELINEAR_BCE_40","r"))
autoencoder.eval()
decoder=pickle.load(open("32_decoder_DAAELINEAR_BCE_40","r"))
xtest=autoencoder.forward(xtest.type(torch.FloatTensor))
xtest=xtest.data.numpy()

#create a panda dataframe
feat_cols = [ 'pixel'+str(i) for i in range(xtest.shape[1]) ]
df = pd.DataFrame(xtest,columns=feat_cols)
df['label'] = ytest
df['label'] = df['label'].apply(lambda i: str(i))
xtest, ytest = None, None

print df.shape
rndperm = np.random.permutation(df.shape[0])

#Plot the graph
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,30):
    ax = fig.add_subplot(3,10,i+1, title='Digit: ' + str(df.loc[rndperm[i],'label']) )
    v=np.array(df.loc[rndperm[i],feat_cols].values).astype(np.float32)
    v=torch.from_numpy(v)
    v=decoder(Variable(v).type(torch.FloatTensor))
    ax.matshow(v.data.numpy().reshape((28,28)))
fig.savefig("latentSpaceessai")
plt.close()
print "finish"

#Apply PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)

chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
chart.save("RAW_DATA_visualizedwithPCA")


#Apply tsne
n_sne = 7000
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

chart2 = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by digit")
chart2.save("RAW_DATA_visualizedwithTSNE")

