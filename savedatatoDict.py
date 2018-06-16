#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pydicom
import pickle

DICT={}
dict={}

def saveDict(root):
    Pathdicom=os.path.join(root,"Patients2")
    for (dirpath, dirnames, filenames) in os.walk(Pathdicom):
        for filename in filenames:
            d=pydicom.read_file(os.path.join(dirpath,filename))
            d_pix_ar = d.pixel_array
            array = np.array(d_pix_ar)
            global dict
            dict[filename]=array
        global DICT
        DICT[dirpath[-7:]]=dict
    del DICT['tients2']
    f=open("dict.pkl","wb")
    pickle.dump(DICT,f)
    f.close()

if __name__=='__main__':
    saveDict("/Users/astrid/Documents/ImplantsProject")
    print len(DICT)