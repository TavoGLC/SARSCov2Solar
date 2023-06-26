#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 02:09:41 2023

@author: tavo
"""

###############################################################################
# Loading packages 
###############################################################################

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import KFold

###############################################################################
# Loading packages 
###############################################################################

def GetSampleLoc(Sample,boundaries):
    cLoc=0
    for k in range(len(boundaries)-1):
        if Sample>=boundaries[k] and Sample<boundaries[k+1]:
            cLoc=k
            break
        
    return cLoc

def GetEqualizedIndex(Data,bins=366,maxCount=100):
    
    np.random.seed(45357487)
  
    cMin,cMax=np.min(Data),np.max(Data)
    boundaries=np.linspace(cMin,cMax,num=bins+1)
  
    SamplesCount=np.zeros(bins)
    indexContainer = []
  
    index=[k for k in range(len(Data))]
    np.random.shuffle(index)
  
    for val in index:
        dataPoint = Data.iloc[val]
        cLoc=GetSampleLoc(dataPoint,boundaries)
      
        if SamplesCount[cLoc]<=maxCount:
            indexContainer.append(val)
            SamplesCount[cLoc]=SamplesCount[cLoc]+1
      
    return indexContainer

###############################################################################
# Loading packages 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/MetaData.csv')
MetaData = MetaData[MetaData['correctdata']==1]
MetaData = MetaData.set_index('id')

reSamplingIndex = GetEqualizedIndex(MetaData['outbreakdays'],maxCount=600,bins=1017)
miniData = MetaData.iloc[reSamplingIndex]

trainInx = miniData.index.values.copy()
np.random.seed(468531)
np.random.shuffle(trainInx)

###############################################################################
# Train Test Kfolds
###############################################################################

outdir = '/media/tavo/storage/biologicalSequences/covidsr2/Folds'

trainDF = pd.DataFrame()
testDF = pd.DataFrame()

folds = KFold(n_splits=5,shuffle=True,random_state=43)

for k,fld in enumerate(folds.split(trainInx)):
    
    train_index, test_index = fld
    
    trainDF['Fold'+str(k)] = trainInx[train_index][0:324133]
    testDF['Fold'+str(k)] = trainInx[test_index][0:81033]
    
valData = MetaData.index.difference(miniData.index)
valData = valData.values.copy()
np.random.seed(76194)
np.random.shuffle(valData)

valDF = pd.DataFrame()

valDF['val'] = valData

trainDF.to_csv(outdir+'/train.csv')
testDF.to_csv(outdir+'/test.csv')
valDF.to_csv(outdir+'/validation.csv')
