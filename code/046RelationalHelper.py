#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2022 Octavio Gonzalez-Lugo 
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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import Normalize

from typing import Sequence,Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from flax.serialization import from_bytes

###############################################################################
# Visualization functions
###############################################################################

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
        self.outConv = nn.Conv(self.outchannels,self.Ksize,padding='SAME',use_bias=False,name='decoder_conv_dec_out')
        
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

###############################################################################
# Loading packages 
###############################################################################

class Helper(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    InputShape: Tuple[int]
    
    depth: int
    out: int
    BatchSize: int
    train: bool = True 
    
    def setup(self):
        self.helperconv =  CoderCONV(self.Units[1::],self.Ksize,self.Strides,self.depth,'convhelper',UpSampling=False,train=self.train)
        
        self.divFactor = 2**(len(self.Units)-1)
        self.targetShape = [val//self.divFactor for val in self.InputShape[0:-1]] + [self.Units[-1]]
        self.localShape = np.prod(np.array(self.targetShape))
        
        self.HelpUnits = [self.localShape,self.localShape//2,self.localShape//4]
        self.helpermlp = CoderMLP(self.HelpUnits,'helpermlp',train=self.train)
        
        self.outh =  nn.Dense(self.out,use_bias=False,name ='helper_out')
        self.outbn = nn.BatchNorm(use_running_average=not self.train,name='helper_bn')
        
    def __call__(self, inputs):
        x = self.helperconv(inputs)
        x = x.reshape((x.shape[0],-1))
        x = self.helpermlp(x)
        x = self.outh(x)
        x = self.outbn(x)
        x = nn.sigmoid(x)
    
        return x

###############################################################################
# Loading packages 
###############################################################################

class ConvHelperVAE(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    InputShape: Tuple[int]
    
    outchannels: int
    depth: int
    BatchSize: int
    auxSize:int
    train: bool = True 
    
    def setup(self):
        self.encoder = CONVEncoder(self.Units,self.Ksize,self.Strides,
                                   self.InputShape,self.depth,self.BatchSize,
                                   self.train)
        self.decoder = CONVDecoder(self.Units[::-1],self.Ksize,self.Strides,
                                   self.InputShape,self.outchannels,
                                   self.depth,self.BatchSize,
                                   self.train)
        self.helper = Helper(self.Units,self.Ksize,self.Strides,
                                   self.InputShape,self.depth,
                                   self.auxSize,self.BatchSize,
                                   self.train)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        hlp = self.helper(recon_x)
        return recon_x, mean, logvar,hlp

##############################################################################
# Data loading 
###############################################################################

batchSize = 64
InputShape = (16,16,16,1)

depth = 2
mainUnits = [4,12,12,4]

auxColumn = ['sunspots','UVB','Vis','NIR','SPdelta','SSTdelta','SATdelta','CFdelta','T03delta']

def VAEModel():
    return ConvHelperVAE(mainUnits,(3,3,3),(2,2,2),InputShape,1,depth,batchSize,len(auxColumn))

##############################################################################
# Data loading 
###############################################################################

basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/Models/RelationalSupervised'

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

helperparams = {'params':state['params']['helper'],'batch_stats':state['batch_stats']['helper']}

def HelperModel(batch):
    return Helper(mainUnits,(3,3,3),(2,2,2),InputShape,depth,len(auxColumn),batchSize,train=False).apply(helperparams, batch)

###############################################################################
# Loading packages 
###############################################################################

bounds = np.linspace(-2.5,2.5, num=65)
xvals = np.zeros(64)

variations = []
outputs = []

for k in range(64):
    yvals = np.linspace(bounds[k],bounds[k+1],num=64)
    walk_vals = np.stack((yvals,xvals)).T
    latentwalk = DecoderModel(walk_vals)
    
    variation = np.array(latentwalk.std(axis=[0,2,3])).ravel()
    variations.append(variation)
    
    targets = HelperModel(latentwalk)
    outputs.append(np.array(targets.mean(axis=0)))

variations = np.vstack(variations)
outputs = np.vstack(outputs)

###############################################################################
# Loading packages 
###############################################################################

cmap = cm.get_cmap('viridis')
normalizer = Normalize(-2.5,2.5)
im = cm.ScalarMappable(norm=normalizer)

fig = plt.figure(figsize=(20,20))
gs = gridspec.GridSpec(nrows=len(auxColumn)+1, ncols=16) 
axs = np.array([fig.add_subplot(gs[k,j]) for k in range(len(auxColumn)) for j in range(16)]).reshape((len(auxColumn),16))

for k in range(len(auxColumn)):
    for j in range(16):
        axs[k,j].scatter(variations[:,j],outputs[:,k],c=bounds[0:-1])
        ImageStyle(axs[k,j])
        axs[-1,j].set_xlabel('Fragment \n'+str(j+1),fontsize=16)
    axs[k,0].set_ylabel(auxColumn[k],fontsize=16)

axmp = fig.add_subplot(gs[-1,:])
axmp.text(0.41, 0.99, 'Genome Variability', size=18, color='black', ha='left', va='top', transform=axmp.transAxes)
cbaxes = axmp.inset_axes([0,0.5,1,0.15])
cbar = plt.colorbar(im,cax=cbaxes, orientation='horizontal')
cbar.set_label('Latent dimension Z0',fontsize=18)

ImageStyle(axmp)

plt.tight_layout()
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr04/data/images/environmentalvar.png',dpi=75,bbox_inches='tight')
plt.close()
