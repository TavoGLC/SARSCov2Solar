#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 02:09:41 2023

@author: tavo
"""

###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd

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

train = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/train.csv')
trainIndex = train['train'].values

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')
MetaData = MetaData.loc[trainIndex]

reSamplingIndex = GetEqualizedIndex(MetaData['outbreakdays'],maxCount=600,bins=1276)
miniData = MetaData.iloc[reSamplingIndex]

trainInx = miniData.index.values.copy()
np.random.seed(88564)
np.random.shuffle(trainInx)

###############################################################################
# Train Test Kfolds
###############################################################################

outdir = '/media/tavo/storage/biologicalSequences/covidsr04/data/folds/BiasedA'

trainDF = pd.DataFrame()
testDF = pd.DataFrame()

folds = KFold(n_splits=5,shuffle=True,random_state=43)

for k,fld in enumerate(folds.split(trainInx)):
    
    train_index, test_index = fld
    
    trainDF['Fold'+str(k)] = trainInx[train_index][0:429410]
    testDF['Fold'+str(k)] = trainInx[test_index][0:107350]
    
trainDF.to_csv(outdir+'/train.csv')
testDF.to_csv(outdir+'/test.csv')