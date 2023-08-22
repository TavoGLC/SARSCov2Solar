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
        self.outConv = nn.Conv(self.outchannels,self.Ksize,padding='SAME',use_bias=False,name='decoder conv_dec_out')
        
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

def LoadBatch(paths):
    
    container = []    
    for pth in paths:
        container.append(np.load(pth))
    container = np.stack(container)
    
    return container.reshape((-1,16,16,16,2))

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
# Loading packages 
###############################################################################

basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/Models/Relational'

Models = ['/flax_model'+str(k) for k in range(5)]

index = 4

with open(basePath+Models[index], "rb") as state_f:
    state = from_bytes(VAEModel, state_f.read())
    state = jax.tree_util.tree_map(jnp.array, state)

rng = random.PRNGKey(0)
localparams = {'params':state['params']['decoder'],'batch_stats':state['batch_stats']['decoder']}

def DecoderModel(batch):
    return CONVDecoder(mainUnits,(3,3,3),(2,2,2),InputShape,1,depth,batchSize,train=False).apply(localparams, batch)

###############################################################################
# Loading packages 
###############################################################################

yvals = np.linspace(-2.5,2.5, num=64)
xvals = np.zeros(64)

walk_vals = np.stack((yvals,xvals)).T

latentwalk = DecoderModel(walk_vals)

###############################################################################
# Loading packages 
###############################################################################

def MakeGenerativePanel(DecoderData,size=(15,20)):
    
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(nrows=21, ncols=16)
    # SARS Cov 2 Genome 
    #Michel, C.J., Mayer, C., Poch, O. et al. Characterization of accessory 
    #genes in coronavirus genomes. Virol J 17, 131 (2020). 
    #https://doi.org/10.1186/s12985-020-01402-1
    
    labels = ['ORF1a','ORF1b','S','ORF3a','ORF3bc','E','M','ORF6','ORF7a','ORF7bc','ORF8','ORFN','ORF9bc ','ORF9cc','ORF10']
    sizes = [13203,8086,3849,828,172,228,669,186,366,130,366,1260,294,222,117]
    csizes = np.cumsum(sizes)
    colors = [plt.cm.viridis(val) for val in np.linspace(0,1,num=len(sizes))]
    np.random.shuffle(colors)
    ax = fig.add_subplot(gs[0:3,:])
    ax.barh('GQ', sizes[0], 0.5, label=labels[0],ec=colors[0],fc=colors[0])
    ax.text(-0.025, 1.1, 'A', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
    
    for k in range(1,len(sizes)):
        ax.barh('GQ', sizes[k], 0.5, label=labels[k],left=csizes[k-1],ec=colors[k],fc=colors[k])
    ax.set_xlim([0,30000])
    ax.legend(loc='upper center',bbox_to_anchor=(0.5, 1.25),ncol=8)
    ImageStyle(ax)
    
    sampledwalk = DecoderData[0::4,:,:,:,0]

    container = []
    for val in sampledwalk:
        container.append(np.hstack(val))
    squared = np.vstack(container)
    
    ax = fig.add_subplot(gs[3:19,:])
    
    ax.imshow(squared)
    ImageStyle(ax)
    ax.set_ylabel('Z0',fontsize=fontsize)
    ax.text(-0.025, 1.01, 'B', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
    '''
    for kk in range(16):
        samp = DecoderData[kk*4]
        for jj,frag in enumerate(samp):
            ax = fig.add_subplot(gs[kk+3,jj])
            #if kk==0 and jj==0:
                
            ax.imshow(frag[:,:,0])
            ImageStyle(ax)
    '''      
    ax = fig.add_subplot(gs[19::,:])
    ax.plot(DecoderData[:,:,:,:,0].std(axis=[0,2,3]),color='navy')
    ax.set_xlabel('Fragment',fontsize=fontsize)
    ax.set_ylabel('Variability',fontsize=fontsize)
    ax.text(-0.025, 0.99, 'C', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
    PlotStyle(ax)
    
MakeGenerativePanel(latentwalk)
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr04/data/images/image_walk.png',dpi=300,bbox_inches='tight')
plt.close()