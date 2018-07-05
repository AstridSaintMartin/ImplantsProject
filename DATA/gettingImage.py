#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import shutil

numberPatients=0
d=1

def getWrightPatients(root, filename,rootExternalVolume):
    #function select patients in the csv file whose patientId appears in the text file
    #and copy the selected files from the external volume to local repository
    
    #Transform text file in list to determine number of patients
    WrighPatientsList=open(os.path.join(root,"filelist.txt"),"r")
    reader=csv.reader(WrighPatientsList)
    PatientId=[int(row[0]) for row in reader]
    global numberPatients
    numberPatients=len(set(PatientId)) #set remove possible duplicates frim the list
    
    # Modify csv file to only keep the files corresponding to the ScanDescription of the wright patients 
    ScanDescription=['SAG_3D_DESS_RIGHT']
    df=pd.read_csv(os.path.join(root,filename))
    af=df[df['ParticipantID'].isin(PatientId)]
    af=af[af['SeriesDescription'].isin(ScanDescription)]
    ImagesPath=af['Folder'].values
    af.to_csv('kneeImplants%d.csv'% d)
    
    
    #copying data from the external volume to the computer
    for imageFile in ImagesPath:
        bf=af[af['Folder']==imageFile]
        #
        #following code get the participant id to name the file
        patientid=str(bf.ParticipantID)[7:16]
        if patientid[0]==" ":
            if patientid[1]==" ":
                patientid=patientid[2:]
            else:
                patientid=patientid[1:len(patientid)-1]
        else:
            patientid=patientid[0:len(patientid)-2]
        #copying starts below
        srcdir=os.path.join(rootExternalVolume,imageFile)
        dstdir=os.path.join(root+"/Patients2",patientid)#create a repository for each patient
        shutil.copytree(srcdir,dstdir)




if __name__=='__main__':
    getWrightPatients("/Users/astrid/Documents/ImplantsProject","contents.0E1.csv","/Volumes/TOSHIBA EXT/OAI")
    print numberPatients