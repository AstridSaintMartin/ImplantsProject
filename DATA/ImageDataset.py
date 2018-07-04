#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd


class KneeDataset(Dataset):
    def __init__(self,csv_file,dict_file, transform=None):
        """
        Args:
        csv_file (string): path to the csv file with PatientsID, stack number and Label
        dict_file (string): path to the pickle file containing the dictionnary with the data
        
        """
        self.frame=pd.read_csv(csv_file)
        self.dict=pickle.load(open(dict_file,"r"))
        self.transform=transform
    
    
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, index):
        row_csv=self.frame.iloc[index]
        label=row_csv["Label"]
        array_of_patient=self.dict[row_csv["PatientsID"]]
        array=array_of_patient[row_csv["StackNumber"]]
        array=torch.from_numpy(array.astype(np.float32))
        return array, label


dset=KneeDataset("DatasetKnee.csv","dictImages64x64.pkl")
dataloader = DataLoader(dset, batch_size=4,shuffle=True)
object=next(iter(dataloader))
print object