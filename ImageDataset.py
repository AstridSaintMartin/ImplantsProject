#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import torch.nn.functional as F

class KneeDataset(Dataset):
    def __init__(self,csv_file,dict_file, transform=None):
        """
        Args:
        csv_file (string): path to the csv file with PatientsID, stack number and Label
        dict_file (string): path to the pickle file containing the dictionnary with the data
        transform : the transform
        
        """
        self.csv_file=csv_file
        self.dict_file=dict_file
        self.transform=transform
        
        
        self.frame=pd.read_csv(csv_file)
        self.dict=pickle.load(open(dict_file,"r"))
        
        
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, index):
        row_csv=self.frame.iloc[index]
        label=row_csv["Label"]
        array_of_patient=self.dict[row_csv["PatientsID"]]
        array=array_of_patient[row_csv["StackNumber"]]
        array=array.astype(np.float32)
        array= np.expand_dims(array, axis=0)
        array=torch.from_numpy(array)
        #print "non normalis",array
        if self.transform is not None:
            array=self.transform(array)
            #print "normalise",array
        print array.size()
        return array, label


normalize = transforms.Normalize(mean=[20], std=[30])
Trans=transforms.Compose([normalize])
dset=KneeDataset(csv_file="DATA/DatasetKnee.csv",dict_file="DATA/dictImages64x64.pkl", transform=Trans)
dataloader = DataLoader(dset, batch_size=1,shuffle=True)
img, label=next(iter(dataloader))
print img
print torch.mean(img)
print torch.std(img)
img=img.numpy()
fig=plt.figure()
plt.imshow(img[0,0,:,:],cmap="gray")
fig.savefig("AVECNormalisation")


