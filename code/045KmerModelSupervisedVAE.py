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

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from itertools import product

from typing import Sequence, Callable

import jax
from jax import lax 
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from flax.serialization import from_bytes

###############################################################################
# Visualization functions
###############################################################################

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
    Axes.xaxis.set_tick_params(labelsize=12)
    Axes.yaxis.set_tick_params(labelsize=12)

Alphabet = ['A','C','T','G']
Labels = []

maxSize = 5
for k in range(1,maxSize):
    
    Labels.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
KmerLabels = [item for sublist in Labels for item in sublist]
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
            x = nn.Dense(feat,use_bias=False,name = self.Name+'_layer_'+str(k))(x)
            x = nn.BatchNorm(use_running_average=not self.train,name = self.name+' norm_'+str(k))(x)
            x = nn.leaky_relu(x)
        return x

class Encoder(nn.Module):
    
    Units: Sequence[int]
    train: bool = True
    
    @nn.compact
    def __call__(self, inputs):
        
        x = inputs
        mlpencoded = Coder(self.Units[1::],'encoder',self.train)(x)
        mean_x = nn.Dense(self.Units[-1], name='mean')(mlpencoded)
        logvar_x = nn.Dense(self.Units[-1], name='logvar')(mlpencoded)
        
        return mean_x, logvar_x

class Decoder(nn.Module):
    
    Units: Sequence[int]
    train: bool = True
    
    def setup(self):
        self.decoder = Coder(self.Units[0:-1],'decoder',self.train)
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = self.decoder(x)
        x = nn.Dense(self.Units[-1],use_bias=False, name='out')(x)
        x = nn.BatchNorm(use_running_average=not self.train,name = 'outnorm')(x)
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
sh = 0.0001
lambda_reg = 10**-9

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * sh * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def dynamics(z, retrieved):
    
    ddt = jnp.diff(z,axis=0)
    diff = ddt - retrieved[0:-1]
    
    return sh * jnp.dot(diff,diff)

class Helper(nn.Module):
    
    HUnits: Sequence[int]
    out: int
    train: bool = True
    
    def setup(self):
        self.helper =  Coder(self.HUnits,'helper',self.train)
        self.outh =  nn.Dense(self.out,use_bias=False,name ='helper_out')
        self.outbn = nn.BatchNorm(use_running_average=not self.train,name='helper_bn')
        
    def __call__(self, inputs):
        x = self.helper(inputs)
        x = self.outh(x)
        x = self.outbn(x)
        x = nn.sigmoid(x)
        
        return x

class Dynamic(nn.Module):
    
    kernel_init: Callable = nn.initializers.constant(0)
    
    def setup(self):
        self.kernel = self.param('kernel_dyn',self.kernel_init,(2,4))
        
    @nn.compact
    def __call__(self, inputs):
        
        stacked = jnp.stack((jnp.ones(inputs.shape[0]),inputs[:,0],inputs[:,1],jnp.multiply(inputs[:,0],inputs[:,1])))
        product = lax.dot(self.kernel,stacked)
        
        return product.T
    
class VAEHelper(nn.Module):
    
    Units: Sequence[int]
    HUnits: Sequence[int]
    out: int
    train: bool = True
    
    def setup(self):
        self.encoder = Encoder(self.Units,self.train)
        self.decoder = Decoder(self.Units[::-1],self.train)
        self.helper =  Helper(self.HUnits,self.out)
        self.dynamic = Dynamic()

    def __call__(self, x, z_rng):
        
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)        
        recon_x = self.decoder(z)
        retrieved = self.dynamic(z)
        help_out = self.helper(recon_x)
        
        return recon_x, mean, logvar, z, help_out, retrieved
    
###############################################################################
# Loading packages 
###############################################################################

mainUnits  = [340,64,2]
hUnits = [256,64,8]
sh = 0.0001

lambda_reg = 10**-9

batchSize = 1024
InputShape = 340

#FullData
epochs = 15

auxColumn = ['sunspots','UVB','Vis','NIR','SPdelta','SSTdelta','SATdelta','CFdelta','T03delta']

##############################################################################
# Data loading 
###############################################################################

Validation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/test.csv')

KmerData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/sequences/KmerDataUpdSmall2023.csv')
KmerData = KmerData.set_index('id')

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')

KmerData = KmerData.loc[Validation['validation']]
MetaData = MetaData.loc[Validation['validation']]

MetaData['SPdelta'] = MetaData['SurfPres_Forecast_A'] - MetaData['SurfPres_Forecast_D']
MetaData['SSTdelta'] = MetaData['SurfSkinTemp_A'] - MetaData['SurfSkinTemp_D']
MetaData['SATdelta'] = MetaData['SurfAirTemp_A'] - MetaData['SurfAirTemp_D']
MetaData['CFdelta'] = MetaData['CloudFrc_A'] - MetaData['CloudFrc_D']
MetaData['T03delta'] = MetaData['TotO3_A'] - MetaData['TotO3_D']
MetaData['OLRdelta'] = MetaData['OLR_A'] - MetaData['OLR_D']
MetaData['COLRdelta'] = MetaData['ClrOLR_A'] - MetaData['ClrOLR_D']

for col in ['UVB','Vis','NIR']:
    std = MetaData[col].std()
    mean = MetaData[col].mean()
    MetaData[col] = [val if 2.5>np.abs((val-mean)/std) else mean for val in MetaData[col]] 

###############################################################################
# Loading packages 
###############################################################################

def VAEHelperModel():
    return VAEHelper(mainUnits,hUnits,len(auxColumn))

def TransformData(Model,Data,rng,Bsize=10000):

    VariationalRep = []
    for k in range(0,len(Data),Bsize):
        localfrag = Data[k:k+Bsize]
        mu,logvar = Model(localfrag)
        varfrag = reparameterize(rng,mu,logvar)
        rng, key = random.split(rng)
        VariationalRep.append(varfrag)
    
    VariationalRep = np.vstack(VariationalRep)
    
    return VariationalRep

###############################################################################
# Loading packages 
###############################################################################

#outbreaksampled
basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/Models/KmersSupervised'

Models = ['/flax_model'+str(k) for k in range(5)]
Scalers = ['/scalerFold'+str(k)+'.joblib' for k in range(5)]

index = 4

with open(basePath+Scalers[index], "rb") as scaler_file:
    Scaler = joblib.load(scaler_file)

with open(basePath+Models[index], "rb") as state_f:
    state = from_bytes(VAEHelperModel, state_f.read())
    state = jax.tree_util.tree_map(jnp.array, state)

rng = random.PRNGKey(0)
KmerDataTR = Scaler.transform(np.array(KmerData))

###############################################################################
# Loading packages 
###############################################################################

encoderparams = {'params':state['params']['encoder'],'batch_stats':state['batch_stats']['encoder']}

def EncoderModel(batch):
    return Encoder(mainUnits,train=False).apply(encoderparams, batch)

toTransform = Scaler.transform(np.array(KmerData))
VarRep = TransformData(EncoderModel,toTransform,rng)

MetaData['dim0'] = VarRep[:,0]
MetaData['dim1'] = VarRep[:,1]

###############################################################################
# Loading packages 
###############################################################################

npoints = 1024
xvls = np.linspace(VarRep[:,0].min(),VarRep[:,0].max(), npoints)
yvls = np.zeros(npoints)
walk_vals = np.stack((xvls,yvls))

decoderparams = {'params':state['params']['decoder'],'batch_stats':state['batch_stats']['decoder']}
hparams = {'params':state['params']['helper'],'batch_stats':state['batch_stats']['helper']}

def DecoderModel(batch):
    return Decoder(mainUnits[::-1],train=False).apply(decoderparams, batch)

def HelperModel(batch):
    return Helper(hUnits,len(auxColumn),train=False).apply(hparams, batch)

latentwalk = DecoderModel(walk_vals.T)
latenth = HelperModel(latentwalk)

###############################################################################
# Loading packages 
###############################################################################

fig = plt.figure(figsize=(25,25))
gs = gridspec.GridSpec(nrows=55, ncols=40)

ax = fig.add_subplot(gs[:,0:20])
ax.scatter(MetaData['dim1'].values,MetaData['dim0'].values,c=MetaData['UVB'].values)
ax.text(-0.025, 1.01, 'A', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)

letters = ['B','C','D']

for nr,jj in enumerate([1,4,8]):
    
    axs = [fig.add_subplot(gs[1+nr+kk+17*nr,jj+20]) for kk in range(17) for jj in range(20)]
    axp = fig.add_subplot(gs[nr+17*nr,20])
    axp.text(-0.025, 1.01, letters[nr], size=25, color='black', ha='left', va='top', transform=axp.transAxes)
    axp.axis('off')

    for k,val in enumerate(KmerLabels):
        axs[k].scatter(latenth[:,jj],latentwalk[:,k],s=4,color='navy')
        axs[k].text(0.01, 0.99, val, size=10, color='black', ha='left', va='top', transform=axs[k].transAxes)
        axs[k].axis('off')

plt.savefig('/media/tavo/storage/biologicalSequences/covidsr04/data/images/image_supervised.png',dpi=300,bbox_inches='tight')
plt.close() 