#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pydicom

def plotImage(root,num_image):
    Pathdicom=os.path.join(root,"Patients/10915212/001")
    d=pydicom.read_file(Pathdicom)
    
    
plotImage("/Users/astrid/Documents/ImplantsProject",1)