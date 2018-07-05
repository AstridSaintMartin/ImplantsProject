#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pydicom

DICT={}
dict={}

def plotImage(root):
    Pathdicom=os.path.join(root,"Patients2")
    for (dirpath, dirnames, filenames) in os.walk(Pathdicom):
        for filename in filenames:
            #
            #create a directory for each patient inside the folder images
            new_dir =os.path.join(root,"Images",dirpath[-7:])
            if not os.path.isdir(new_dir):
                os.makedirs(new_dir)
            #transform dicom image into numpy array
            d=pydicom.read_file(os.path.join(dirpath,filename))
            d_pix_ar = d.pixel_array
            array = np.array(d_pix_ar)
            global dict
            dict[filename]=array
            figure=plt.figure()
            plt.imshow(array, interpolation='nearest', cmap="gray")
            figure.savefig(os.path.join(new_dir,filename)+".png")
            plt.close()
        global DICT
        DICT[dirpath[-7:]]=dict
    

if __name__=='__main__':
    plotImage("/Users/astrid/Documents/ImplantsProject")
    print len(DICT)
    print len(dict)

