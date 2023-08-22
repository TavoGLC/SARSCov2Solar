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

import scipy.stats

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
    Name: str 
    train: bool = True 
    
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,feat in enumerate(self.Units):
            x = nn.Dense(feat,use_bias=False,name = self.Name+' layer_'+str(k))(x)
            x = nn.BatchNorm(use_running_average=not self.train,name = self.Name+' norm_'+str(k))(x)
            x = nn.leaky_relu(x)
        return x

class Encoder(nn.Module):
    
    Units: Sequence[int]
    train: bool = True
    
    @nn.compact
    def __call__(self, inputs):
        
        x = inputs
        mlpencoded = Coder(self.Units[1::],'encodermlp',self.train)(x)
        mean_x = nn.Dense(self.Units[-1], name='mean')(mlpencoded)
        logvar_x = nn.Dense(self.Units[-1], name='logvar')(mlpencoded)
        
        return mean_x, logvar_x

class Decoder(nn.Module):
    
    Units: Sequence[int]
    train: bool = True
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        decoded_1 = Coder(self.Units[0:-1],'decodermlp',self.train)(x)
        out = nn.Dense(self.Units[-1],use_bias=False, name='out')(decoded_1)
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
    train: bool = True
    
    def setup(self):
        self.encoder = Encoder(self.Units,self.train)
        self.decoder = Decoder(self.Units[::-1],self.train)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

mainUnits  = [340,64,2]

def VAEModel():
    return VAE(mainUnits)

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
# Data selection
###############################################################################

KmerData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/sequences/KmerDataUpdSmall2023.csv')
KmerData = KmerData.set_index('id')

Validation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/test.csv')

###############################################################################
# Data selection
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')

#FullData
KmerData = KmerData.loc[Validation['validation']]
MetaData = MetaData.loc[Validation['validation']]

MetaData['SPdelta'] = MetaData['SurfPres_Forecast_A'] - MetaData['SurfPres_Forecast_D']
MetaData['SSTdelta'] = MetaData['SurfSkinTemp_A'] - MetaData['SurfSkinTemp_D']
MetaData['SATdelta'] = MetaData['SurfAirTemp_A'] - MetaData['SurfAirTemp_D']
MetaData['CFdelta'] = MetaData['CloudFrc_A'] - MetaData['CloudFrc_D']
MetaData['T03delta'] = MetaData['TotO3_A'] - MetaData['TotO3_D']

###############################################################################
# Loading packages 
###############################################################################

#completedata
#basePath = r'/media/tavo/storage/biologicalSequences/covidsr04/data/Models/KmersFull'

#outbreaksampled
basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/Models/KmersBiased'

Models = ['/flax_model'+str(k) for k in range(5)]
Scalers = ['/scalerFold'+str(k)+'.joblib' for k in range(5)]

index = 4

with open(basePath+Scalers[index], "rb") as scaler_file:
    Scaler = joblib.load(scaler_file)

with open(basePath+Models[index], "rb") as state_f:
    state = from_bytes(VAEModel, state_f.read())
    state = jax.tree_util.tree_map(jnp.array, state)

rng = random.PRNGKey(0)
KmerDataTR = Scaler.transform(np.array(KmerData))
localparams = {'params':state['params']['encoder'],'batch_stats':state['batch_stats']['encoder']}

def EcoderModel(trainparams,batch):
    return Encoder(mainUnits,train=False).apply(trainparams, batch)

VariationalRep = []
Bsize = 80000
for k in range(0,len(KmerDataTR),Bsize):
    localfrag = KmerDataTR[k:k+Bsize]
    mu,logvar = EcoderModel(localparams,localfrag)
    varfrag = reparameterize(rng,mu,logvar)
    rng, key = random.split(rng)
    VariationalRep.append(varfrag)

VariationalRep = np.vstack(VariationalRep)

del KmerDataTR

#Complete
#MetaData['ldim0'] = VariationalRep[:,0]
#MetaData['ldim1'] = VariationalRep[:,1]    

#Sampled
MetaData['ldim0'] = VariationalRep[:,1]
MetaData['ldim1'] = VariationalRep[:,0]    


###############################################################################
# Loading packages 
###############################################################################

order = np.argsort(VariationalRep[:,1])
ordval = np.array([val for val in order if np.abs(VariationalRep[val,1])<0.0125])
RepData = VariationalRep[ordval]
ord_x = np.argsort(RepData[:,0])

indexOrder = ordval[ord_x]
indexIds = KmerData.iloc[indexOrder].index

del KmerData,VariationalRep

###############################################################################
# Loading packages 
###############################################################################

MiniData = MetaData.loc[indexIds].copy()
MiniData['date'] = pd.to_datetime(MiniData['date'])

###############################################################################
# Loading packages 
###############################################################################

featlistg =['daylength','daylengthd10','ldim0','ldim1',
            'dayofyear','SPdelta','SSTdelta','UVB','Vis','NIR','CFdelta',
            'T03delta','sunspots','outbreakdays']

plotnames =['daylength','daylengthd10','outbreakdays','dayofyear','SPdelta',
            'SSTdelta','sunspots','UVB','Vis','NIR','CFdelta','T03delta']

correlation  = MetaData[featlistg].corr()

###############################################################################
# Loading packages 
###############################################################################

def MakeDimReductionPlot(Data,column,colname,ax):
    localPlot = ax.scatter(Data['ldim0'],Data['ldim1'],c=Data[column],alpha=0.2,cmap='viridis')
    cbar = plt.colorbar(localPlot,ax=ax)
    ax.set_xlabel('Z0',fontsize=fontsize)
    ax.set_ylabel('Z1',fontsize=fontsize)
    cbar.solids.set(alpha=1)
    cbar.set_label(colname,fontsize=fontsize)

###############################################################################
# Loading packages 
###############################################################################

fig = plt.figure(figsize=(25,25))
gs = gridspec.GridSpec(nrows=6, ncols=6) 

axs0 = fig.add_subplot(gs[0:2,0:3])

MakeDimReductionPlot(MetaData,'outbreakdays','Days Since the Initial Outbreak',axs0)
PlotStyle(axs0)
axs0.text(0.01, 0.99, 'A', size=25, color='black', ha='left', va='top', transform=axs0.transAxes)

axs1 = fig.add_subplot(gs[2:4,0:3])
MakeDimReductionPlot(MetaData,'dayofyear','Day of the Year (DOY)',axs1)
PlotStyle(axs1)
axs1.text(0.01, 0.99, 'B', size=25, color='black', ha='left', va='top', transform=axs1.transAxes)

axs2 = fig.add_subplot(gs[4:6,0:3])
MakeDimReductionPlot(MetaData,'daylengthd10','Sunshine Duration Rate of Change (SDRC)',axs2)
PlotStyle(axs2)
axs2.hlines(y=0.0025,xmin=MetaData['ldim0'].min(),xmax=MetaData['ldim0'].max(),linewidth=1, color='navy',alpha=0.5)
axs2.hlines(y=-0.0025,xmin=MetaData['ldim0'].min(),xmax=MetaData['ldim0'].max(),linewidth=1, color='navy',alpha=0.5)
axs2.text(0.01, 0.99, 'C', size=25, color='black', ha='left', va='top', transform=axs2.transAxes)

###############################################################################
# Loading packages 
###############################################################################

axs = [fig.add_subplot(gs[k,3:6]) for k in range(6)]
xlabs = ['SD','SDRC','OD','DOY','SPdelta','SSTdelta','NS','UVB','Vis','NIR','CFdelta','T03delta']
correlation['ldim0'].abs()[plotnames].plot.bar(color='navy',ax=axs[0])
axs[0].set_ylabel('Absolute Pearson Correlation')
axs[0].set_xticks(np.arange(len(plotnames)))
axs[0].set_xticklabels(xlabs,rotation=45)
axs[0].text(0.01, 0.99, 'D', size=25, color='black', ha='left', va='top', transform=axs[0].transAxes)
PlotStyle(axs[0])

###############################################################################
# Loading packages 
###############################################################################

n_rolling = 100
features = ['dayofyear','daylengthd10','sunspots','Vis','Length'] 
names = ['DOY','SDRC', 'NS', 'Visible Radiation', 'Sequence Length']
letters = ['E','F','G','H','I','J','K']
rollingdata = MiniData[features].rolling(n_rolling)

for k,val in enumerate(features):
    
    rollingmean = rollingdata[val].apply(np.median)
    rollingerror = scipy.stats.t.ppf(1.95/2,n_rolling-1) * rollingdata[val].sem()
    rollingplus = rollingmean + rollingerror
    rollingminus = rollingmean - rollingerror
    
    rollingmean.plot(ax=axs[k+1],label=val,color='navy',legend=False)
    rollingplus.plot(ax=axs[k+1],color='black',alpha=0.5,legend=False)
    rollingminus.plot(ax=axs[k+1],color='black',alpha=0.5,legend=False)
    axs[k+1].fill_between(np.arange(rollingmean.shape[0]), np.array(rollingminus), np.array(rollingplus),  color='black', alpha=0.051)
    
    localRange = 0.25*(rollingmean.max()-rollingmean.min())
    axs[k+1].set_ylim([rollingmean.min()-localRange, rollingmean.max()+localRange])
    axs[k+1].set_ylabel(names[k])
    axs[k+1].text(0.01, 0.99, letters[k], size=25, color='black', ha='left', va='top', transform=axs[k+1].transAxes)
    PlotStyle(axs[k+1])
    
plt.tight_layout()
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr04/data/images/image_fold_'+str(index)+'.png',dpi=75,bbox_inches='tight')
plt.close()
