#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2021 Octavio Gonzalez-Lugo 

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

import numpy as np
import multiprocessing as mp

from Bio import SeqIO
from io import StringIO

###############################################################################
# Global definitions
###############################################################################

GlobalDirectory = r"/media/tavo/storage/biologicalSequences/covid/seqs sep 2022/"
seqDataDir = GlobalDirectory+'sequences.fasta'

sequencesFrags = GlobalDirectory + 'splitted/'

MaxCPUCount = int(0.8*mp.cpu_count())
CANONICALALPHABET = np.sort(['A','C','T','G'])

###############################################################################
# Sequence processing
###############################################################################

#Wrapper function to ask if the number of unique elements in a sequence is equal to 4 
def CanonicalAlphabetQ(sequence):
    uniques = np.sort(np.unique(sequence))
    if len(uniques)==4:
        disc = uniques==CANONICALALPHABET
        if all(disc):
            return True
        else:
            return False
    else:
        return False
    
#Wrapper function to ask if the number of unique elements in a sequence is equal to 4 
def SizeQ(sequence):
    if len(sequence.seq)>25000:
        return True
    else:
        return False

###############################################################################
# Sequence Filtering Functions
###############################################################################

def GetFilteredSeqsIndex(Sequences):
    '''
    Iterates over a series of DNA sequences to chek if the only avaliable bases 
    are A,C,G,T. 

    Parameters
    ----------
    Sequences : list
        list of strings, Contains the DNA sequences

    Returns
    -------
    canonical : list
        Index of the sequences that contains only four bases .
    noncanonical : list
        Index of the sequences that contains more than four bases. A different 
        character on a DNA sequence means that it could be one  of 2 or more bases 
        at that position

    '''
            
    localPool=mp.Pool(MaxCPUCount)
    canonicalResponce=localPool.map(CanonicalAlphabetQ,[val for val in Sequences])
    localPool.close()
    
    localPool=mp.Pool(MaxCPUCount)
    sizeResponce=localPool.map(SizeQ,[val for val in Sequences])
    localPool.close()
    
    canonical=[]
    
    for k,disc in enumerate(zip(canonicalResponce,sizeResponce)):
        
        if disc[0] and disc[1]:
            canonical.append(k)
        
    return canonical

###############################################################################
# Sequence Loading functions
###############################################################################

def BatchIterator(Iterator, BatchSize):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < BatchSize:
            try:
                entry = next(Iterator)
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch

Sequences = SeqIO.parse(open(seqDataDir), "fasta")
total = 0
for i, batch in enumerate(BatchIterator(Sequences, 35000)):
    filename = sequencesFrags + "group_%i.fasta" % (i + 1)
    sequenceIndex = GetFilteredSeqsIndex(batch)
    newbatch = [batch[k] for k in sequenceIndex]
    total = total + len(newbatch)
    
    with open(filename, "w") as handle:
        count = SeqIO.write(newbatch, handle, "fasta")
    print("Wrote %i records to %s" % (count, filename))
    print(total)
