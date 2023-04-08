#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 31 22:10:27 2022

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
import matplotlib.patches as patches

import scipy.stats

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

DataPath = r"/media/tavo/storage/biologicalSequences/covid/datasets"
DataDir = DataPath+'/KmerDataUpdSmall.csv'
KmerData = pd.read_csv(DataDir)
KmerData = KmerData.set_index('id')

###############################################################################
# Data selection
###############################################################################

headers = ['id'] + KmerLabels 

KmerData = pd.read_csv(DataDir,usecols=headers)
KmerData = KmerData.set_index('id')

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/MetaData.csv')
MetaData = MetaData[MetaData['correctdata']==1]
MetaData = MetaData.set_index('id')

KmerData = KmerData.loc[MetaData.index]

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

if index==1:    
    MetaData['ldim0'] = VariationalRep[:,1]
    MetaData['ldim1'] = VariationalRep[:,0]
else:
    MetaData['ldim0'] = VariationalRep[:,0]
    MetaData['ldim1'] = VariationalRep[:,1]


###############################################################################
# Loading packages 
###############################################################################

order = np.argsort(VariationalRep[:,1])
ordval = np.array([val for val in order if np.abs(VariationalRep[val,1])<0.0025])
RepData = VariationalRep[ordval]
ord_x = np.argsort(RepData[:,0])

indexOrder = ordval[ord_x]
indexIds = KmerData.iloc[indexOrder].index

del KmerData,VariationalRep

###############################################################################
# Loading packages 
###############################################################################

MiniData = MetaData.loc[indexIds].copy()

###############################################################################
# Loading packages 
###############################################################################

basePath = '/media/tavo/storage/biologicalSequences/covid/seqs sep 2022/single'
seqPaths = [basePath+'/'+fil+'.fasta' for fil in MiniData.index]

seqs = [GetSeq(val) for val in seqPaths]

maxSeq = max([len(val.seq) for val in seqs])

localDict = {}

for k,val in enumerate(Alphabet):
    localDict[val]=k+1

CodedSeqs = [[localDict[val] for val in sal.seq] + [0 for _ in range(maxSeq-len(sal.seq))] for sal in seqs]
CodedSeqs = np.array(CodedSeqs)

lst = ['TTTAAAA','TTTAAAC','TTTAAAT','TTTAAAG']

CodedSeqs0 = [[1 if sal.seq[k:k+len(lst[0])] in lst  else 0 for k in range(len(sal.seq)-len(lst[0]))] + [0 for _ in range(maxSeq-len(sal.seq))] for sal in seqs]
CodedSeqs0 = np.array(CodedSeqs0)

lst1 = ['TTTAAAA']

CodedSeqs1 = [[1 if sal.seq[k:k+len(lst[0])] in lst1  else 0 for k in range(len(sal.seq)-len(lst[0]))] + [0 for _ in range(maxSeq-len(sal.seq))] for sal in seqs]
CodedSeqs1 = np.array(CodedSeqs1)

###############################################################################
# Loading packages 
###############################################################################

fig,axs =  plt.subplots(6,3,figsize=(17,8))

for k in range(6):    
    
    axs[k,0].imshow(CodedSeqs[:,k*5000:(k+1)*5000],aspect=0.45,cmap='Blues_r')
    axs[k,0].axes.xaxis.set_ticks([])
    axs[k,0].axes.yaxis.set_ticks([])
    axs[k,1].imshow(CodedSeqs0[:,k*5000:(k+1)*5000],aspect=0.45,cmap='Blues_r')
    axs[k,1].axes.xaxis.set_ticks([])
    axs[k,1].axes.yaxis.set_ticks([])
    axs[k,2].imshow(CodedSeqs1[:,k*5000:(k+1)*5000],aspect=0.45,cmap='Blues_r')
    axs[k,2].axes.xaxis.set_ticks([])
    axs[k,2].axes.yaxis.set_ticks([])

rect = patches.Rectangle((1400, 350), 150, 350, linewidth=1, edgecolor='r', facecolor='none')
axs[1,2].add_patch(rect)

rect = patches.Rectangle((3050, 350), 150, 350, linewidth=1, edgecolor='r', facecolor='none')
axs[4,2].add_patch(rect)

axs[0,0].text(0.01, 0.99, 'A', size=25, color='black', ha='left', va='top', transform=axs[0,0].transAxes)
axs[0,1].text(0.01, 0.99, 'B', size=25, color='black', ha='left', va='top', transform=axs[0,1].transAxes)
axs[0,2].text(0.01, 0.99, 'C', size=25, color='black', ha='left', va='top', transform=axs[0,2].transAxes)

plt.tight_layout()
plt.savefig('/media/tavo/storage/images/image_sequences.png',dpi=75,bbox_inches='tight')
