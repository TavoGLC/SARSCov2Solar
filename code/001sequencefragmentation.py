#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 02:25:46 2023

@author: tavo
"""

import os
import multiprocessing as mp

from Bio import SeqIO
from io import StringIO

###############################################################################
# Loading packages 
###############################################################################
#Wrapper function to load the sequences
def GetSeqs(Dir):
    
    cDir=Dir
    
    with open(cDir) as file:
        
        seqData=file.read()
        
    Seq=StringIO(seqData)
    SeqList=list(SeqIO.parse(Seq,'fasta'))
    
    return SeqList

def GetFragment(sequence):
    
    seq = str(sequence.seq)
    start = seq.find('ATG')
    stopc = ['TAA','TAG','TGA']
    stops = [1 if seq[k:k+3] in stopc else 0 for k in range(len(seq)-3)]
    ends = [k for k,val in enumerate(stops) if val==1]
    
    return seq[start:ends[-1]+3]

MaxCPUCount = int(0.8*mp.cpu_count())

###############################################################################
# Loading packages 
###############################################################################

def GetSeqFragment(Sequences):
    
    localPool=mp.Pool(MaxCPUCount)
    fragments=localPool.map(GetFragment,[val for val in Sequences])
    localPool.close()
    
    ids = [val.id for val in Sequences]
        
    return fragments,ids

def SaveFiles(sequences,names,outpath):
    for sq,nme in zip(sequences,names):
        with open(outpath+nme+'.txt','w') as f:
            f.write(sq)
            
###############################################################################
# Loading packages 
###############################################################################

filepath = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/splitted/'
outputdir = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/single/'

seqfiles = os.listdir(filepath)
counter = 0

for fil in seqfiles:
    
    loop_path = filepath+fil
    seqs = GetSeqs(loop_path)
    a,b = GetSeqFragment(seqs)
    SaveFiles(a,b,outputdir)
    counter = counter + len(seqs)
    print(counter)
