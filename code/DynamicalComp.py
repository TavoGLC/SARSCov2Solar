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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import Normalize

import scipy.stats

from itertools import product

from typing import Sequence

import jax
import jax.numpy as jnp

from jax import random
from flax import linen as nn
from flax.serialization import from_bytes

from scipy.integrate import odeint

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

def ImageStyle(Axes): 
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
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.set_xticks([])
    Axes.set_yticks([])

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

mainUnits  = [340,64,2]

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

###############################################################################
# Loading packages 
###############################################################################

def dynamicsf(y,t,coefs):
    
    vec = np.array([y[0],y[1],y[0]*y[1]])
    dydt = np.dot(coefs[:,1::],vec) + coefs[:,0]

    return dydt

###############################################################################
# Data selection
###############################################################################

DataPath = r"/media/tavo/storage/biologicalSequences/covid/datasets"
DataDir = DataPath+'/KmerDataUpdSmall.csv'
KmerData = pd.read_csv(DataDir)
KmerData = KmerData.set_index('id')

#OutbreakSampled
foldsPath = r'/media/tavo/storage/biologicalSequences/covidsr2/Folds'

#FullData
#foldsPath = r'/media/tavo/storage/biologicalSequences/covidsr2/FoldsFull'

Validation = pd.read_csv(foldsPath+'/validation.csv')

###############################################################################
# Data selection
###############################################################################

headers = ['id'] + KmerLabels 

KmerData = pd.read_csv(DataDir,usecols=headers)
KmerData = KmerData.set_index('id')

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/MetaDataNASA.csv')
MetaData = MetaData.set_index('id')

#OutbreakSampled
KmerData = KmerData.loc[Validation['val'][0:100000]]
MetaData = MetaData.loc[Validation['val'][0:100000]]

#FullData
#KmerData = KmerData.loc[Validation['validation']]
#MetaData = MetaData.loc[Validation['validation']]

MetaData = MetaData[MetaData['correctdata']==1]
KmerData = KmerData.loc[MetaData.index]

MetaData['SPdelta'] = MetaData['SurfPres_Forecast_A'] - MetaData['SurfPres_Forecast_D']
MetaData['SSTdelta'] = MetaData['SurfSkinTemp_A'] - MetaData['SurfSkinTemp_D']
MetaData['SATdelta'] = MetaData['SurfAirTemp_A'] - MetaData['SurfAirTemp_D']
MetaData['CFdelta'] = MetaData['CloudFrc_A'] - MetaData['CloudFrc_D']
MetaData['T03delta'] = MetaData['TotO3_A'] - MetaData['TotO3_D']

wlnames = ['UVA','UVB','UVC','Vis','NIR','SWIR']
wdata = pd.read_csv('/media/tavo/storage/sunspots/solarinterpolated.csv')
wdata['date'] = pd.to_datetime(wdata['date'])
wdata = wdata.set_index('date')
wdata = wdata.rolling(7).apply(np.median)

for val in wlnames:    
    MetaData[val] = np.array(wdata[val].loc[MetaData['date']])

###############################################################################
# Loading packages 
###############################################################################

#outbreaksampled
basePath = '/media/tavo/storage/biologicalSequences/covidsr2/Models/Kmers'

#completedata
#basePath = r'/media/tavo/storage/biologicalSequences/covidsr2/Models/KmersFull'

fig = plt.figure(figsize=(25,25))
gs = gridspec.GridSpec(nrows=5, ncols=5) 

cmap = cm.get_cmap('viridis')
normalizer = Normalize(MetaData['UVB'].min(),MetaData['UVB'].max())
im = cm.ScalarMappable(norm=normalizer)

ax0 = fig.add_subplot(gs[:,0])

cbaxes = ax0.inset_axes([0.425,0,0.15,1])
cbar = plt.colorbar(im,cax=cbaxes, orientation='vertical')
cbar.set_label('UVB',fontsize=16)

ImageStyle(ax0)

axs = [fig.add_subplot(gs[j,k+1]) for j in range(5) for k in range(4)]
axs = np.array(axs).reshape(5,4)

for i in range(5):
    
    with open(basePath+'/scalerFold'+str(i)+'.joblib', "rb") as scaler_file:
        Scaler = joblib.load(scaler_file)

    with open(basePath+'/flax_model'+str(i), "rb") as state_f:
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
        rng, key = random.split(rng)
        VariationalRep.append(varfrag)

    VariationalRep = np.vstack(VariationalRep)
    axs[i,0].scatter(VariationalRep[:,0],VariationalRep[:,1],c=MetaData['UVB'],norm=normalizer,alpha=0.1)
    axs[i,0].set_xlabel('Z0',fontsize=fontsize)
    axs[i,0].set_ylabel('Z1',fontsize=fontsize)
    PlotStyle(axs[i,0])

#dynamics 
basePath = '/media/tavo/storage/biologicalSequences/covidsr2/Models/KmersDynamics'

for i in range(5):
    
    with open(basePath+'/scalerFold'+str(i)+'.joblib', "rb") as scaler_file:
        Scaler = joblib.load(scaler_file)

    with open(basePath+'/flax_model'+str(i), "rb") as state_f:
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
        rng, key = random.split(rng)
        VariationalRep.append(varfrag)

    VariationalRep = np.vstack(VariationalRep)
    axs[i,1].scatter(VariationalRep[:,0],VariationalRep[:,1],c=MetaData['UVB'],norm=normalizer,alpha=0.1)
    axs[i,1].set_xlabel('Z0',fontsize=fontsize)
    axs[i,1].set_ylabel('Z1',fontsize=fontsize)
    PlotStyle(axs[i,1])
    
    def locDyn(y,t):
        return dynamicsf(y,t,np.array(state['params']['dynamic']['kernel_dyn']))
    
    yA0 = [0.01,10000]
    yB0 = [10000,0.01]
    t = np.linspace(0, 12, 1000)
        
    solA = odeint(locDyn, yA0, t)
    solB = odeint(locDyn, yB0, t)
    
    ynormA0 = (solA[:,0]-solA[:,0].min())/(solA[:,0].max()-solA[:,0].min())
    ynormA1 = (solA[:,1]-solA[:,1].min())/(solA[:,1].max()-solA[:,1].min())
    
    if ynormA0.max()!=1 or ynormA1.max()!=1:
        ynormA0 = np.ones(1000)
        ynormA1 = np.ones(1000)
            
    axs[i,2].plot(t,ynormA0,color='red')
    axs[i,2].plot(t,ynormA1,color='black')
    axs[i,2].text(7.5,0.85,'Z0',fontsize=15,color='red')
    axs[i,2].text(8.5,0.85,'<<<',fontsize=15,alpha=0.5)
    axs[i,2].text(10.25,0.85,'Z1',fontsize=15,color='black')
    axs[i,2].set_xlabel('Time (days)',fontsize=fontsize)
    axs[i,2].set_ylabel('Amount',fontsize=fontsize)
    axs[i,2].set_ylim([0, 1])
    PlotStyle(axs[i,2])
    
    ynormB0 = (solB[:,0]-solB[:,0].min())/(solB[:,0].max()-solB[:,0].min())
    ynormB1 = (solB[:,1]-solB[:,1].min())/(solB[:,1].max()-solB[:,1].min())
    
    if ynormB0.max()!=1 or ynormB1.max()!=1:
        ynormB0 = np.ones(1000)
        ynormB1 = np.ones(1000)
    
    axs[i,3].plot(t,ynormB0,color='red')
    axs[i,3].plot(t,ynormB1,color='black')
    axs[i,3].text(7.5,0.85,'Z0',fontsize=15,color='red')
    axs[i,3].text(8.5,0.85,'>>>',fontsize=15,alpha=0.5)
    axs[i,3].text(10.25,0.85,'Z1',fontsize=15,color='black')
    axs[i,3].set_xlabel('Time (days)',fontsize=fontsize)
    axs[i,3].set_ylabel('Amount',fontsize=fontsize)
    axs[i,3].set_ylim([0, 1])
    PlotStyle(axs[i,3])

letters = ['A','B','C','D']
for jj in range(4):
    axs[0,jj].text(0.01, 0.99, letters[jj], size=25, color='black', ha='left', va='top', transform=axs[0,jj].transAxes)

plt.tight_layout()
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr2/images/image_dyn_models.png',dpi=75,bbox_inches='tight')
plt.close()
