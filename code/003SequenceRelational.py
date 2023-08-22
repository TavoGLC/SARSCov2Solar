#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 02:16:07 2023

@author: tavo
"""

##############################################################################
# Loading packages 
###############################################################################

import os 
import numpy as np
import multiprocessing as mp

from itertools import product
from collections import Counter
from numpy import linalg as LA

###############################################################################
# Global definitions
###############################################################################

MaxCPUCount=int(0.80*mp.cpu_count())

###############################################################################
# Blocks
###############################################################################

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 4
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])

###############################################################################
# Sequence K-mer generating functions
###############################################################################

def ReadFile(path):    
    with open(path) as f:
        lines = f.readlines()
    return str(lines[0])

###############################################################################
# Sequence K-mer generating functions
###############################################################################

def SplitString(String,ChunkSize):
    '''
    Split a string ChunkSize fragments using a sliding windiow

    Parameters
    ----------
    String : string
        String to be splitted.
    ChunkSize : int
        Size of the fragment taken from the string .

    Returns
    -------
    Splitted : list
        Fragments of the string.

    '''
      
    if ChunkSize==1:
        Splitted=[val for val in String]
    
    else:
        nCharacters=len(String)
        Splitted=[String[k:k+ChunkSize] for k in range(nCharacters-ChunkSize)]
        
    return Splitted

###############################################################################
# Sequence Graphs Functions
###############################################################################

def MakeAdjacencyList(processedSequence,Block,skip=0):
    
    CharactersToLocation = dict([(val,k) for k,val in enumerate(Block)])
    
    x,y = [],[]
    
    for k in range(len(processedSequence)-skip-1):
        backFragment = processedSequence[k]
        forwardFragment = processedSequence[k+skip+1]
            
        if backFragment in Block and forwardFragment in Block:
            x.append(CharactersToLocation[backFragment])
            y.append(CharactersToLocation[forwardFragment])
            
    return x,y

def RelationalSkip(Sequence,Block,skip=0):
    
    D12 = np.zeros((len(Block),len(Block)))
    currentMatrix = np.zeros((len(Block),len(Block)))
    fragmentSize=len(Block[0])
    
    processedSequence=SplitString(Sequence,fragmentSize)
    x,y = MakeAdjacencyList(processedSequence,Block,skip=skip)
    
    pairs = [val for val in zip(x,y)]
    counts = Counter(pairs)
    
    for ky in counts.keys():
        currentMatrix[ky] = counts[ky]
        
    currentMatrix = currentMatrix + currentMatrix.T
    
    
    for k,val in enumerate(currentMatrix.sum(axis=0)):
        D12[k,k] = 1/np.sqrt(2*val)
    
    currentMatrix = np.dot(D12,currentMatrix).dot(D12)
    w,v = LA.eig(currentMatrix)
    norm = LA.norm(w)
    
    return currentMatrix/norm

def MakeEncoding(Sequence,Block,skip):
    
    relationalForward = RelationalSkip(Sequence,Block,skip=skip)
    relationalForward = (relationalForward-relationalForward.min())/(relationalForward.max()-relationalForward.min())
    
    return relationalForward.reshape((len(Block),len(Block),1))

def Make4DEncoding(Sequence,Block=Blocks[1],fragments=16):
    
    Sequence = Sequence
    step = len(Sequence)//fragments
    container = np.zeros(shape=(fragments,len(Block),len(Block),1))
    toStringSequence = str(Sequence)
    
    for k in range(fragments):
        localSequence = toStringSequence[k*step:(k+1)*step]
        currentEncoding = MakeEncoding(localSequence,Block,1)
        container[k,:,:,:] = currentEncoding 
        
    return container 

def GetDataParallel(DataBase,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.map(Function, [val for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Sequence Graphs Functions
###############################################################################

matrixData = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/relational/'

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
    
    data = GetDataParallel(loadedSeqs,Make4DEncoding)
    
    counter = counter + len(currentPaths)
    print(counter)    
    
    for nme, db in zip(names,data):
        np.save(matrixData+nme, db)
