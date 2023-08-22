#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 02:09:41 2023

@author: tavo
"""

###############################################################################
# Loading packages 
###############################################################################

import pandas as pd

from sklearn.model_selection import KFold

###############################################################################
# Train Test Kfolds
###############################################################################

outdir = '/media/tavo/storage/biologicalSequences/covidsr04/data/folds/Full'
train = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/train.csv')
trainInx = train['train'].values

###############################################################################
# Train Test Kfolds
###############################################################################

trainDF = pd.DataFrame()
testDF = pd.DataFrame()

folds = KFold(n_splits=5,shuffle=True,random_state=43)

for k,fld in enumerate(folds.split(trainInx)):
    
    train_index, test_index = fld
    
    trainDF['Fold'+str(k)] = trainInx[train_index][0:1128700]
    testDF['Fold'+str(k)] = trainInx[test_index][0:282170]
    
trainDF.to_csv(outdir+'/train.csv')
testDF.to_csv(outdir+'/test.csv')
