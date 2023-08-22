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

from typing import Sequence, Tuple

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
class CoderCONV(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    
    depth: int
    Name: str 
    UpSampling: bool = True
    train: bool = True
                    
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,val in enumerate(self.Units):
            finalDepth = self.depth-k+1
            if finalDepth<=1:
                finalDepth = 1
            
            for ii in range(finalDepth):
                x = nn.Conv(val,self.Ksize,padding='SAME',use_bias=False,
                            name=self.Name+' conv_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.Name+' conv_norm_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
                
            if self.UpSampling:
                x = nn.ConvTranspose(val,self.Ksize,padding='SAME',
                                     strides=self.Strides,use_bias=False,
                                     name=self.Name+' convUp_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.Name+' conv_normUp_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
            else:
                x = nn.Conv(val,self.Ksize,padding='SAME',strides=self.Strides,
                            use_bias=False,
                            name=self.Name+' convDn_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.Name+' conv_normDn_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
            
        return x

class CoderMLP(nn.Module):
    
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

###############################################################################
# Loading packages 
###############################################################################

class EncoderMLP(nn.Module):
    
    Units: Sequence[int]
    train: bool = True
    
    def setup(self):
        self.encoder = CoderMLP(self.Units[1::],'encoderlmlp',train=self.train)
        self.mean = nn.Dense(self.Units[-1], name='mean')
        self.logvar = nn.Dense(self.Units[-1], name='logvar')
    
    def __call__(self, inputs):
        
        x = inputs
        mlpencoded = self.encoder(x)
        mean_x = self.mean(mlpencoded)
        logvar_x = self.logvar(mlpencoded)
        
        return mean_x, logvar_x

class DecoderMLP(nn.Module):
    
    Units: Sequence[int]
    train: bool = True
    
    def setup(self):
        self.decoder = CoderMLP(self.Units[0:-1],'decodermlp',train=self.train)
        self.out = nn.Dense(self.Units[-1],use_bias=False, name='out')
        self.outnorm = nn.BatchNorm(use_running_average=not self.train,name = 'outnorm')
    
    def __call__(self, inputs):
        x = inputs
        decoded_1 = self.decoder(x)
        
        out =self.out(decoded_1)
        out = self.outnorm(out)
        out = nn.leaky_relu(out)
        
        return out

###############################################################################
# Loading packages 
###############################################################################

class CONVEncoder(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    InputShape: Tuple[int]
    
    depth: int
    BatchSize: int
    train: bool = True 
    
    def setup(self):
        
        self.localConv = CoderCONV(self.Units,self.Ksize,self.Strides,self.depth,'convencoder',UpSampling=False,train=self.train)
        self.divFactor = 2**(len(self.Units)-1)
        
        self.targetShape = [val//self.divFactor for val in self.InputShape[0:-1]] + [self.Units[-1]]
        
        self.localShape = np.prod(np.array(self.targetShape))
        self.EncUnits = [self.localShape,self.localShape//4,self.localShape//16,2]
        self.localEncoder = EncoderMLP(self.EncUnits,train=self.train)
        
    def __call__(self,inputs):
        
        x = inputs
        x = self.localConv(x)
        x = x.reshape((x.shape[0],-1))
        mean_x,logvar_x = self.localEncoder(x)
        
        return mean_x,logvar_x

class CONVDecoder(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    InputShape: Tuple[int]
    
    outchannels: int
    depth: int
    BatchSize: int
    train: bool = True 
    
    def setup(self):
        
        self.localConv = CoderCONV(self.Units[1::],self.Ksize,self.Strides,self.depth,'convdecoder',UpSampling=True,train=self.train)
        self.divFactor = 2**(len(self.Units)-1)
        
        self.finalShape = [self.BatchSize]+[val//self.divFactor for val in self.InputShape[0:-1]] + [self.Units[-1]]
        
        self.localShape = np.prod(np.array(self.finalShape[1::]))
        self.DecUnits = [2,self.localShape//16,self.localShape//4,self.localShape]
        
        self.localDecoder = DecoderMLP(self.DecUnits,train=self.train)
        
        self.outnorm = nn.BatchNorm(use_running_average=not self.train,name = 'outnorm')
        self.outConv = nn.Conv(self.outchannels,self.Ksize,padding='SAME',use_bias=False,name=self.name+' conv_dec_out')
        
    #@nn.compact
    def __call__(self,inputs):
        
        x = inputs
        x = self.localDecoder(x)
        x = jnp.reshape(jnp.array(x),self.finalShape)
        x = self.localConv(x)
        x = self.outConv(x)
        x = self.outnorm(x)
        x = nn.sigmoid(x)
        
        return x

###############################################################################
# Loading packages 
###############################################################################

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std

class ConvVAE(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    InputShape: Tuple[int]
    
    outchannels: int
    depth: int
    BatchSize: int
    name: str 
    train: bool = True 
    
    def setup(self):
        self.encoder = CONVEncoder(self.Units,self.Ksize,self.Strides,
                                   self.InputShape,self.depth,self.BatchSize,
                                   self.train)
        self.decoder = CONVDecoder(self.Units[::-1],self.Ksize,self.Strides,
                                   self.InputShape,self.outchannels,
                                   self.depth,self.BatchSize,
                                   self.train)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

###############################################################################
# Loading packages 
###############################################################################

def MakeAnnealingWeights(epochs,cycles,scale=1):
    '''
    Parameters
    ----------
    epochs : int
        min size of the array to return.
    cycles : int
        number of annealing cycles.
    scale : float, optional
        scales the annealing weights. The default is 1.

    Returns
    -------
    array
        annealing weights.

    '''
    pointspercycle = epochs//cycles
    AnnealingWeights = 1*(1/(1+np.exp(-1*np.linspace(-10,10,num=pointspercycle))))
    
    for k in range(cycles-1):
        AnnealingWeights = np.append(AnnealingWeights,1*(1/(1+np.exp(-1*np.linspace(-10,10,num=pointspercycle+10)))))
        
    return scale*AnnealingWeights

def LoadBatch(paths):
    
    container = []    
    for pth in paths:
        container.append(np.load(pth))
    container = np.stack(container)
    
    return container.reshape((-1,16,16,16,1))

def DataLoader(datadirs,batchsize):
    for k in range(0,len(datadirs),batchsize):
        paths = datadirs[k:k+batchsize]
        yield LoadBatch(paths)
        
###############################################################################
# Loading packages 
###############################################################################

batchSize = 64
InputShape = (16,16,16,1)

depth = 2
mainUnits = [4,12,12,4]

def VAEModel():
    return ConvVAE(mainUnits,(3,3,3),(2,2,2),InputShape,2,depth,batchSize,'test')
 
def TransformData(Model,Data,Bsize=10000):

    VariationalRep = []
    rng = random.PRNGKey(0)
    
    for batch in DataLoader(Data,Bsize):
        
        mu,logvar = Model(batch)
        varfrag = reparameterize(rng,mu,logvar)
        VariationalRep.append(varfrag)
        rng, key = random.split(rng)
    
    VariationalRep = np.vstack(VariationalRep)
    
    return VariationalRep

###############################################################################
# Data selection
###############################################################################

Validation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/test.csv')

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')
MetaData = MetaData.loc[Validation['validation']]

MetaData['SPdelta'] = MetaData['SurfPres_Forecast_A'] - MetaData['SurfPres_Forecast_D']
MetaData['SSTdelta'] = MetaData['SurfSkinTemp_A'] - MetaData['SurfSkinTemp_D']
MetaData['SATdelta'] = MetaData['SurfAirTemp_A'] - MetaData['SurfAirTemp_D']
MetaData['CFdelta'] = MetaData['CloudFrc_A'] - MetaData['CloudFrc_D']
MetaData['T03delta'] = MetaData['TotO3_A'] - MetaData['TotO3_D']

dataPath = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/relational'
valData = np.array([dataPath+'/'+val+'.npy' for val in MetaData.index])

###############################################################################
# Loading packages 
###############################################################################

basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/Models/Relational'

Models = ['/flax_model'+str(k) for k in range(5)]

index = 4

with open(basePath+Models[index], "rb") as state_f:
    state = from_bytes(VAEModel, state_f.read())
    state = jax.tree_util.tree_map(jnp.array, state)

rng = random.PRNGKey(0)
localparams = {'params':state['params']['encoder'],'batch_stats':state['batch_stats']['encoder']}

def EncoderModel(batch):
    return CONVEncoder(mainUnits,(3,3,3),(2,2,2),InputShape,depth,batchSize,train=False).apply(localparams, batch)


VariationalRep = TransformData(EncoderModel,valData,1*batchSize)

MetaData['ldim0'] = VariationalRep[:,0]
MetaData['ldim1'] = VariationalRep[:,1]

###############################################################################
# Loading packages 
###############################################################################

order = np.argsort(VariationalRep[:,1])
ordval = np.array([val for val in order if np.abs(VariationalRep[val,1])<0.0125])
RepData = VariationalRep[ordval]
ord_x = np.argsort(RepData[:,0])

indexOrder = ordval[ord_x]
indexIds = MetaData.iloc[indexOrder].index

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
