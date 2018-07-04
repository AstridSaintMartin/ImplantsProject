#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import shutil
import pickle
import copy
import cv2

def getPatient():
    #the file classes contain the patient id for which a stl file could be generated (it wasn't possible for the 197 patients 
    #that were in the original database
    #the file classes gives a hash integer to modelize the class that we call abusively 'Label'
    WrightPatient=open("Classes.txt","r")
    reader=csv.reader(WrightPatient)
    dictionary={}
    dict=pickle.load(open("dict.pkl","r"))
    dictSTL={}
    rownumber=0
    for row in reader:
        if rownumber !=0:
            if row[0] in dict.keys():
                dictionary[int(row[0])]=int(row[1])
                dictSTL[int(row[0])]=dict[row[0]]
        rownumber+=1
    file=open("dicCLasses.pkl","w")
    filename=open("dicImages.pkl","w")
    pickle.dump(dictionary,file)
    pickle.dump(dictSTL,filename)    




def transformto64():
    dict=pickle.load(open("dicImages.pkl","r"))
    classes=pickle.load(open("dicClasses.pkl","r"))
    dictionnary={}
    littledic={}
    
    #create the csv file that we are going to use in the dataset class
    file=open("DatasetKnee.csv","w")
    columns=["PatientsID","StackNumber","Label"]
    writer=csv.DictWriter(file,fieldnames=columns)
    writer.writeheader()
    
    for key in dict.keys():
        stacksdic=dict[key]
        for KEY in stacksdic.keys():
            littledic[KEY]=cv2.resize(stacksdic[KEY],(64,64))
            writer.writerow({"PatientsID": key,"StackNumber":KEY, "Label":classes[key]})
        dictionnary[key]=littledic
    pickle.dump(dictionnary, open("dictImages64x64.pkl","w"))


#getPatient()
#transformto64()
df=pd.read_csv(open("DatasetKnee.csv","r"))
print df
a= df.iloc[0]
print a
print a["Label"]
