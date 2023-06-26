#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 02:34:59 2023

@author: tavo
"""

import os 
import numpy as np
import multiprocessing as mp

from Bio import SeqIO
from io import StringIO

##############################################################################
# Global definitions
###############################################################################

GlobalDirectory = r"/media/tavo/storage/biologicalSequences/covid/seqs sep 2022"
sequencesFrags = GlobalDirectory + '/splitted'

matrixData = '/media/tavo/storage/biologicalSequences/covid/datasets/OHEMD'

MaxCPUCount=int(0.80*mp.cpu_count())

##############################################################################
# Sequence Loading functions
###############################################################################

#Wrapper function to load the sequences
def GetSeqs(Dir):
    
    cDir=Dir
    
    with open(cDir) as file:
        
        seqData=file.read()
        
    Seq=StringIO(seqData)
    SeqList=list(SeqIO.parse(Seq,'fasta'))
    
    return SeqList  

###############################################################################
# Blocks
###############################################################################

Alphabet = ['A','C','T','G']

TokenDictionary = {}

for k,val in enumerate(Alphabet):
    currentVec = [0 for j in range(len(Alphabet))]
    currentVec[k] = 1
    TokenDictionary[val]=currentVec

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

fragmentsDirs = [sequencesFrags + '/' + dr for dr in os.listdir(sequencesFrags)]
counter = 0

for blk in fragmentsDirs:
    
    Container = GetSeqs(blk)
    
    print(counter)

    names = [seq.id for seq in Container]
    data = GetDataParallel(Container,MakeSequenceEncoding)
    counter = counter + len(Container)
    
    for nme, db in zip(names,data):
        np.save(matrixData+'/'+nme, db)
        