#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import shutil

numberPatients=0
d=0

def getWrightPatients(root, filename,rootExternalVolume):
    #select patients in the csv file whose patientId appears in the text file
    
    #Transform text file in list to determine number of patients
    WrighPatientsList=open(os.path.join(root,"filelist.txt"),"r")
    reader=csv.reader(WrighPatientsList)
    PatientId=[int(row[0]) for row in reader]
    global numberPatients
    numberPatients=len(set(PatientId)) #set remove possible duplicates frim the list
    
    # Modify csv file to only keep the files of the wright patients
    ScanDescription=['SAG_3D_DESS_RIGHT','SAG_3D_DESS_LEFT']
    df=pd.read_csv(os.path.join(root,filename))
    af=df[df['ParticipantID'].isin(PatientId)]
    af=af[af['SeriesDescription'].isin(ScanDescription)]
    ImagesPath=af['Folder'].values
    af.to_csv('kneeImplants%d.csv'% d)
    
    
    #copying data from the external volume to the computer
    #for imageFile in ImagesPath:
        #srcdir=os.path.join(rootExternalVolume,imageFile)
        #dstdir=os.path.join(root,str(imageFile)[-8:])
        #shutil.copytree(srcdir,dstdir)




if __name__=='__main__':
    getWrightPatients("/Users/astrid/Documents/ImplantsProject","contents.OE1.csv","/Volumes/TOSHIBA\ EXT/")
    print numberPatients