#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2023 Octavio Gonzalez-Lugo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
@author: Octavio Gonzalez-Lugo
"""

###############################################################################
# Loading packages 
###############################################################################

import os 
import numpy as np
import pandas as pd
import multiprocessing as mp

from itertools import product

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

def UniqueToDictionary(UniqueElements):
    '''
    Creates a dictionary that takes a Unique element as key and return its 
    position in the UniqueElements array
    Parameters
    ----------
    UniqueElements : List,array
        list of unique elements.

    Returns
    -------
    localDictionary : dictionary
        Maps element to location.

    '''
    
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

###############################################################################
# Sequences as graphs. 
###############################################################################

def CountUniqueElements(UniqueElements,String):
    '''
    Calculates the frequency of the unique elements in a splited or 
    processed string. Returns a list with the frequency of the 
    unique elements. 
    
    Parameters
    ----------
    UniqueElements : array,list
        Elements to be analized.
    String : strting
        Sequence data.
    Processed : bool, optional
        Controls if the sring is already splitted or not. The default is False.
    Returns
    -------
    localCounter : array
        Normalized frequency of each unique fragment.
    '''
    
    nUnique = len(UniqueElements)
    localCounter = [0 for k in range(nUnique)]
    
    ProcessedString = SplitString(String,len(UniqueElements[0]))    
        
    nSeq = len(ProcessedString)
    UniqueDictionary = UniqueToDictionary(UniqueElements)
    
    for val in ProcessedString:
        
        if val in UniqueElements:
            
            localPosition=UniqueDictionary[val]
            localCounter[localPosition]=localCounter[localPosition]+1
            
    localCounter=[val/nSeq for val in localCounter]
        
    return localCounter


def CountUniqueElementsByBlock(Sequences,UniqueElementsBlock):
    '''
    
    Parameters
    ----------
    Sequences : list, array
        Data set.
    UniqueElementsBlock : list,array
        Unique element collection of different fragment size.
    config : bool, optional
        Controls if the sring is already splitted or not. The default is False.
    Returns
    -------
    Container : array
        Contains the frequeny of each unique element.
    '''
    
    Container=np.array([[],[]])
    
    for k,block in enumerate(UniqueElementsBlock):
        
        countPool=mp.Pool(MaxCPUCount)
        currentCounts=countPool.starmap(CountUniqueElements, [(block,val)for val in Sequences])
        countPool.close()
        
        if k==0:
            Container=np.array(currentCounts)
        else:
            Container=np.hstack((Container,currentCounts))
            
    return Container

###############################################################################
# Blocks
###############################################################################

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 5
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
###############################################################################
# Blocks
###############################################################################

MaxCPUCount=int(0.85*mp.cpu_count())
seqsdir = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/single/'
files = os.listdir(seqsdir)

paths = [seqsdir+val for val in files]

chunkSize = 10000
blocksContainer = []
names = []

for k in range(0,len(paths),chunkSize):
    currentPaths = paths[k:k+chunkSize]
    names = names + [val[len(seqsdir):-4] for val in currentPaths]
    loadedSeqs = [ReadFile(sal) for sal in currentPaths]
    blocksContainer.append(CountUniqueElementsByBlock(loadedSeqs,Blocks))
    print(len(names))

Kmers = np.vstack(blocksContainer)
 
KmerDF = pd.DataFrame()
headers = [val for li in Blocks for val in li]
KmerDF = pd.DataFrame(Kmers,columns=headers)
KmerDF['id'] = names
KmerDF = KmerDF.set_index('id')
KmerDF.to_csv('/media/tavo/storage/biologicalSequences/covidsr04/sequences/KmerDataUpd2023.csv')

###############################################################################
# Blocks
###############################################################################

container = []
for blk in Blocks:
    
    innerContainer = []
    vals = KmerDF[blk].values
    for val in vals:
        norm = (val - val.min())/(val.max() - val.min())
        innerContainer.append(norm)
    container.append(np.array(innerContainer).astype(np.float16))
    
container = np.hstack(container)

KmerDFSmall = pd.DataFrame()

KmerDFSmall = pd.DataFrame(container,columns=headers)
KmerDFSmall['id'] = names
KmerDFSmall = KmerDFSmall.set_index('id')
KmerDFSmall.to_csv('/media/tavo/storage/biologicalSequences/covidsr04/sequences/KmerDataUpdSmall2023.csv')
