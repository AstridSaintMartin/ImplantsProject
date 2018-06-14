#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

numberPatients=0

def getWrightPatients(root, filename):
    #select patients in the csv file whose patientId appears in the text file
    
    #Transform text file in list to determine number of patients
    WrighPatientsList=open(os.path.join(root,"filelist.txt"),"r")
    reader=csv.reader(WrighPatientsList)
    PatientId=[int(row[0]) for row in reader]
    global numberPatients
    numberPatients=len(set(PatientId)) #set remove possible duplicates frim the list
    
    # Modify csv file to only keep the files of the wright patients
    ScanDescription=['SAG_3D_DESS_RIGHT']
    df=pd.read_csv(os.path.join(root,filename))
    af=df[df['ParticipantID'].isin(PatientId)]
    af=af[af['SeriesDescription'].isin(ScanDescription)]
    ImagesPath=af['Folder'].values
    print ImagesPath
    af.to_csv('kneeImplants.csv')
    





getWrightPatients("/Users/astrid/Documents/ImplantsProject","contents.4G1.csv")
print numberPatients