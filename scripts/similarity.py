#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:01:53 2023

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

from scipy import linalg
from sklearn.covariance import ledoit_wolf
from scipy.spatial import distance as ds
from sklearn.metrics import mean_squared_error

from Bio import SeqIO
from io import StringIO
from itertools import product

from typing import Sequence

import jax
import jax.numpy as jnp

from jax import random
from flax import linen as nn
from flax.serialization import from_bytes


###############################################################################
# Visualization functions
###############################################################################
fontsize = 16

def PlotStyle(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=fontsize)
    Axes.yaxis.set_tick_params(labelsize=fontsize)

###############################################################################
# Loading packages 
###############################################################################

class Coder(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True 
    
    def setup(self):
        self.layers = [nn.Dense(feat,use_bias=False,name = self.name+' layer_'+str(k)) for k,feat in enumerate(self.Units)]
        self.norms = [nn.BatchNorm(use_running_average=not self.train,name = self.name+' norm_'+str(k)) for k,feat in enumerate(self.Units)]
        
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,block in enumerate(zip(self.layers,self.norms)):
            lay,norm = block
            x = lay(x)
            x = norm(x)
            x = nn.relu(x)
        return x

class Encoder(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.encoder = Coder(self.Units[1::],self.name,self.train)
        self.mean = nn.Dense(self.Units[-1], name='mean')
        self.logvar = nn.Dense(self.Units[-1], name='logvar')
    
    @nn.compact
    def __call__(self, inputs):
        
        x = inputs
        mlpencoded = self.encoder(x)
        mean_x = self.mean(mlpencoded)
        logvar_x = self.logvar(mlpencoded)
        
        return mean_x, logvar_x

class Decoder(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.decoder = Coder(self.Units[0:-1],self.name,self.train)
        self.out = nn.Dense(self.Units[-1],use_bias=False, name='out')
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        decoded_1 = self.decoder(x)
        
        out =self.out(decoded_1)
        out = nn.BatchNorm(use_running_average=not self.train,name = 'outnorm')(out)
        out = nn.sigmoid(out)
        
        return out

###############################################################################
# Loading packages 
###############################################################################

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std

class VAE(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.encoder = Encoder(self.Units,self.name+'encoder',self.train)
        self.decoder = Decoder(self.Units[::-1],self.name+'decoder',self.train)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

mainUnits  = [340,170,85,21,5,2]

def VAEModel():
    return VAE(mainUnits,'test')

###############################################################################
# Loading packages 
###############################################################################

Alphabet = ['A','C','T','G']
Labels = []

maxSize = 5
for k in range(1,maxSize):
    
    Labels.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
KmerLabels = [item for sublist in Labels for item in sublist]

def GetSeq(path):

    with open(path) as file:
        
        seqData=file.read()
        
    Seq=StringIO(seqData)
    SeqList=list(SeqIO.parse(Seq,'fasta'))
    
    return SeqList[0]

###############################################################################
# Data selection
###############################################################################

KmerTranscripts = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/referenceTranscrips/TranscriptsUpdSmall.csv')
KmerTranscripts = KmerTranscripts.set_index('id')

transcriptsdir = '/media/tavo/storage/biologicalSequences/covid/datasets/referenceTranscrips/'

DataPath = r"/media/tavo/storage/biologicalSequences/covid/datasets"
DataDir = DataPath+'/KmerDataUpdSmall.csv'
KmerData = pd.read_csv(DataDir)
KmerData = KmerData.set_index('id')

###############################################################################
# Loading packages 
###############################################################################

basePath = '/media/tavo/storage/biologicalSequences/covid/models/Model0jax'

Models = ['/flax_model'+str(k) for k in range(5)]
Scalers = ['/scalerFold'+str(k)+'.joblib' for k in range(5)]

index = 0

with open(basePath+Scalers[index], "rb") as scaler_file:
    Scaler = joblib.load(scaler_file)

with open(basePath+Models[index], "rb") as state_f:
    state = from_bytes(VAEModel, state_f.read())
    state = jax.tree_util.tree_map(jnp.array, state)

rng = random.PRNGKey(0)
KmerDataTR = Scaler.transform(np.array(KmerData))
localparams = {'params':state['params']['testencoder'],'batch_stats':state['batch_stats']['testencoder']}

def EcoderModel(trainparams,batch):
    return Encoder(mainUnits,'testencoder',train=False).apply(trainparams, batch)

VariationalRep = []
Bsize = 80000
for k in range(0,len(KmerDataTR),Bsize):
    localfrag = KmerDataTR[k:k+Bsize]
    mu,logvar = EcoderModel(localparams,localfrag)
    varfrag = reparameterize(rng,mu,logvar)
    VariationalRep.append(varfrag)

VariationalRep = np.vstack(VariationalRep)

del KmerDataTR

###############################################################################
# Loading packages 
###############################################################################

order = np.argsort(VariationalRep[:,1])
ordval = np.array([val for val in order if np.abs(VariationalRep[val,1])<0.0025])
RepData = VariationalRep[ordval]
ord_x = np.argsort(RepData[:,0])

indexOrder = ordval[ord_x]
indexIds = KmerData.iloc[indexOrder].index

###############################################################################
# Loading packages 
###############################################################################

MiniData = KmerData.loc[indexIds].copy()

del KmerData

###############################################################################
# Loading packages 
###############################################################################

XA = np.array(MiniData)

CovA, _ = ledoit_wolf(XA)
IconvA = linalg.inv(CovA)
Xmean = XA.mean(axis=0).reshape(-1,1).T

XB = np.array(KmerTranscripts)

Y = ds.cdist(XB, Xmean, 'mahalanobis', VI=IconvA)
Y = Y.min(axis=1)

KmerTranscripts['dist'] = Y

hdata = np.histogram(Y,bins=1500)
threshold = hdata[1][np.argmax(hdata[0])]

MiniTranscripts = KmerTranscripts[KmerTranscripts['dist']<threshold].copy()
scaled = Scaler.transform(np.array(MiniTranscripts[KmerLabels]))

def VAEModel(trainparams,batch):
    return VAE(mainUnits,'test',train=False).apply(trainparams, batch,rng)

reconstruction,_,_ = VAEModel(state,scaled)

rec_error = np.array([mean_squared_error(sc,rec) for sc,rec in zip(scaled,reconstruction)])

MiniTranscripts['rec'] = rec_error

hrec_data = np.histogram(rec_error,bins=1500)
rec_threshold = hrec_data[1][np.argmax(hrec_data[0])]

MiniTranscripts = MiniTranscripts[MiniTranscripts['rec']<rec_threshold]

outfile = transcriptsdir+'fold_out_'+str(index)+'.txt'

with open(outfile, 'w') as f:
    for line in MiniTranscripts.index:
        f.write(line)
        f.write('\n')
        
        
