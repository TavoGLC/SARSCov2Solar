#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 02:34:59 2023

@author: tavo
"""

import os 
import numpy as np
import multiprocessing as mp

###############################################################################
# Blocks
###############################################################################

MaxCPUCount=int(0.80*mp.cpu_count())

Alphabet = ['A','C','T','G']

TokenDictionary = {}

for k,val in enumerate(Alphabet):
    currentVec = [0 for j in range(len(Alphabet))]
    currentVec[k] = 1
    TokenDictionary[val]=currentVec

###############################################################################
# Sequence K-mer generating functions
###############################################################################

def ReadFile(path):    
    with open(path) as f:
        lines = f.readlines()
    return str(lines[0])

###############################################################################
# Blocks
###############################################################################

def MakeSequenceEncoding(Sequence):
    
    stringFrags = [val for val in Sequence]
    nToAdd = (32*32*32) - len(stringFrags)
    toAdd = [[0,0,0,0] for k in range(nToAdd)]
    encoded = [TokenDictionary[val] for val in stringFrags] + toAdd    
    encoded = np.array(encoded).reshape((32,32,32,4))
    
    return encoded.astype(np.int8)

###############################################################################
# Blocks
###############################################################################

def GetDataParallel(DataBase,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.map(Function, [(val )for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Sequence Graphs Functions
###############################################################################

matrixData = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/OHE/'

seqsdir = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/single/'
files = os.listdir(seqsdir)

paths = [seqsdir+val for val in files]

chunkSize = 10000
blocksContainer = []
names = []
counter = 0

for k in range(0,len(paths),chunkSize):
    
    currentPaths = paths[k:k+chunkSize]
    names = [val[len(seqsdir):-4] for val in currentPaths]
    loadedSeqs = [ReadFile(sal) for sal in currentPaths]
    
    data = GetDataParallel(loadedSeqs,MakeSequenceEncoding)
    
    counter = counter + len(currentPaths)
    print(counter)    
    
    for nme, db in zip(names,data):
        np.save(matrixData+nme, db)

        