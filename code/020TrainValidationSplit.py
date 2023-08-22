#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 01:03:48 2023

@author: tavo
"""

###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

###############################################################################
# Loading packages 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')

Xtrain, Xtest, _, _ = train_test_split(MetaData['id'].values, MetaData['id'].values, test_size=0.1, random_state=42)

np.random.seed(46832)
np.random.shuffle(Xtrain)

train = pd.DataFrame()
train['train'] = Xtrain
train.to_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/train.csv',index=False)

np.random.seed(467462)
np.random.shuffle(Xtest)

test = pd.DataFrame()
test['validation'] = Xtest
test.to_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/test.csv',index=False)