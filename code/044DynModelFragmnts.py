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
import time
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split

from typing import Sequence, Callable

import jax
import optax
from jax import lax 
from jax import random
import jax.numpy as jnp
from flax import linen as nn

import matplotlib.gridspec as gridspec

from scipy.integrate import odeint

###############################################################################
# Visualization functions
###############################################################################
fontsize=16

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
    Name: str 
    train: bool = True 
    
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,feat in enumerate(self.Units):
            x = nn.Dense(feat,use_bias=False,name = self.name+' layer_'+str(k))(x)
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

class Dynamic(nn.Module):
    
    kernel_init: Callable = nn.initializers.constant(0)
    
    def setup(self):
        self.kernel = self.param('kernel_dyn',self.kernel_init,(2,4))
        
    @nn.compact
    def __call__(self, inputs):
        
        stacked = jnp.stack((jnp.ones(inputs.shape[0]),inputs[:,0],inputs[:,1],jnp.multiply(inputs[:,0],inputs[:,1])))
        product = lax.dot(self.kernel,stacked)
        
        return product.T
    
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
        self.dynamic = Dynamic()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        retrieved = self.dynamic(z)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar, z, retrieved

###############################################################################
# Loading packages 
###############################################################################

mainUnits  = [340,64,2]
sh = 0.0001

lambda_reg = 10**-9

batchSize = 1024
InputShape = 340

###############################################################################
# Data selection
###############################################################################

KmerData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/sequences/KmerDataUpdSmall2023.csv')
KmerData = KmerData.set_index('id')

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData['qut'] = pd.qcut(MetaData['daylength'],8)

cats = MetaData['qut'].unique()
catsDict = {}
for k,val in enumerate(cats):
    catsDict[val]=k

MetaData['cat'] = [catsDict[val] for val in MetaData['qut']]
MetaData = MetaData.set_index('id')

KmerData = KmerData.loc[MetaData.index]

###############################################################################
# Loading packages 
###############################################################################
@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * sh * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def dynamics(z, retrieved):
    
    ddt = jnp.diff(z,axis=0)
    diff = ddt - retrieved[0:-1]
    
    return sh * jnp.dot(diff,diff)

def MainLoss(Model,params,batchStats,z_rng ,batch):
  
    block, newbatchst = Model().apply({'params': params, 'batch_stats': batchStats}, batch, z_rng,mutable=['batch_stats'])
    recon_x, mean, logvar, z, retrieved = block
    
    regloss = 0
    for w in jax.tree_util.tree_leaves(params):
        regloss = regloss + lambda_reg*jnp.linalg.norm(w,ord=1)#jnp.mean(jnp.abs(w))
        
    kld_loss = kl_divergence(mean, logvar).mean(axis=-1)
    loss_value = optax.l2_loss(recon_x, batch).mean(axis=-1)
    dyn_loss = dynamics(z,retrieved).mean() #+ (10**-6)*jnp.linalg.norm(params['dynamic']['kernel_dyn'][:,1::],ord=1)
    
    total_loss = loss_value.mean() + kld_loss.mean() + regloss + dyn_loss
    
    return total_loss,newbatchst['batch_stats']


def TrainModel(TrainData,TestData,Loss,params,batchStats,rng,epochs=10,batch_size=64,lr=0.005):
    
    totalSteps = epochs*(TrainData.shape[0]//batch_size) + epochs
    stepsPerCycle = totalSteps//4
    
    esp = [{"init_value":lr/10, 
            "peak_value":(lr)/((k+1)), 
            "decay_steps":int(stepsPerCycle*0.75), 
            "warmup_steps":int(stepsPerCycle*0.25), 
            "end_value":lr/10} for k in range(8)]
    
    Scheduler = optax.sgdr_schedule(esp)
    localOptimizer = optax.adam(learning_rate=Scheduler)
    optState = localOptimizer.init(params)
    
    @jax.jit
    def step(params,batchStats ,optState, z_rng, batch):
        
        (loss_value,batchStats), grads = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch)
        updates, optState = localOptimizer.update(grads, optState, params)
        params = optax.apply_updates(params, updates)
        
        return params,batchStats, optState, loss_value
    
    @jax.jit
    def getloss(params,batchStats, z_rng, batch):
        (loss_value,_), _ = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch)
        return loss_value
    
    trainloss = []
    testloss = []
    
    for epoch in range(epochs):
        
        st = time.time()
        batchtime = []
        losses = []
        
        for k in range(0,TrainData.shape[0],batch_size):
    
            stb = time.time()
            batch = TrainData[k:k+batch_size]
            
            rng, key = random.split(rng)
            params,batchStats ,optState, lossval = step(params,batchStats,optState,key,batch)
            losses.append(lossval)
            batchtime.append(time.time()-stb)
        
        valloss = []
        for i in range(0,TestData.shape[0],batch_size):
            rng, key = random.split(rng)
            val_batch = TestData[i:i+batch_size]
            valloss.append(getloss(params,batchStats,key,val_batch))
        
        mbatch = 1000*np.mean(batchtime)
        meanloss = np.mean(losses)
        meanvalloss = np.mean(valloss)
        
        trainloss.append(meanloss)
        testloss.append(meanvalloss)
        np.random.shuffle(TrainData)
    
        end = time.time()
        output = 'Epoch = '+str(epoch) + ' Time per epoch = ' + str(round(end-st,3)) + 's  Time per batch = ' + str(round(mbatch,3)) + 'ms' + ' Train Loss = ' + str(meanloss) +' Test Loss = ' + str(meanvalloss)
        print(output)
        
    return trainloss,testloss,params,batchStats

###############################################################################
# Loading packages 
###############################################################################

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

def dynamicsf(y,t,coefs):
    
    vec = np.array([y[0],y[1],y[0]*y[1]])
    dydt = np.dot(coefs[:,1::],vec) + coefs[:,0]

    return dydt
    
###############################################################################
# Loading packages 
###############################################################################

std = MetaData['UVB'].std()
mean = MetaData['UVB'].mean()

MetaData['UVB'] = [val if 2.5>np.abs((val-mean)/std) else mean for val in MetaData['UVB']] 

cmap = cm.get_cmap('viridis')
normalizer = colors.Normalize(MetaData['UVB'].min(),MetaData['UVB'].max())
im = cm.ScalarMappable(norm=normalizer)
intervals = MetaData['qut'].unique()

fig = plt.figure(figsize=(27,25))
gs = gridspec.GridSpec(nrows=3, ncols=9) 

ax0 = fig.add_subplot(gs[:,0])

cbaxes = ax0.inset_axes([0.425,0,0.15,1])
cbar = plt.colorbar(im,cax=cbaxes, orientation='vertical')
cbar.set_label('UVB',fontsize=16)

ImageStyle(ax0)

axs = [fig.add_subplot(gs[j,k+1]) for j in range(3) for k in range(8)]
axs = np.array(axs).reshape(3,8)

for k,ys in enumerate(intervals):
    
    np.random.seed(128)
    data_index = MetaData[MetaData['qut']==ys].index
    miniData = MetaData.loc[data_index]
    
    kmer_data = KmerData.loc[data_index].values
    
    Xtrain, Xtest, _, _ = train_test_split(kmer_data, np.arange(len(kmer_data)), test_size=0.15, random_state=42)
    
    scaler = pr.MinMaxScaler()
    scaler.fit(Xtrain)
    
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    
    def VAEModel():
        return VAE(mainUnits,'test')
    
    def loss(params,batchStats,z_rng ,batch):
        return MainLoss(VAEModel,params,batchStats,z_rng ,batch)
    
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    
    init_data = jnp.ones((batchSize, InputShape), jnp.float32)
    initModel = VAEModel().init(key, init_data, rng)
    
    params0 = initModel['params']
    batchStats = initModel['batch_stats']

    trloss,tstloss,params0,batchStats = TrainModel(Xtrain,Xtest,loss,params0,
                                        batchStats,rng,lr=0.05,epochs=20,
                                        batch_size=batchSize)
    
    state = {'params':params0,'batch_stats':batchStats}
    localparams = {'params':state['params']['encoder'],'batch_stats':state['batch_stats']['encoder']}

    def EncoderModel(batch):
        return Encoder(mainUnits,train=False).apply(localparams, batch)
    
    finalData = scaler.transform(kmer_data)
    
    rep = TransformData(EncoderModel,finalData,rng,Bsize=int(len(finalData)/10))
    
    axs[0,k].scatter(rep[:,0],rep[:,1],c=MetaData['UVB'].loc[data_index],norm=normalizer,alpha=0.15)
    axs[0,k].set_xlabel('Z0',fontsize=fontsize)
    axs[0,k].set_ylabel('Z1',fontsize=fontsize)
    PlotStyle(axs[0,k])
    
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
            
    axs[1,k].plot(t,ynormA0,color='red')
    axs[1,k].plot(t,ynormA1,color='black')
    axs[1,k].text(7.5,0.85,'Z0',fontsize=14,color='red')
    axs[1,k].text(8.5,0.85,'<<<',fontsize=14,alpha=0.5)
    axs[1,k].text(10.25,0.85,'Z1',fontsize=14,color='black')
    axs[1,k].set_xlabel('Time (days)',fontsize=fontsize)
    axs[1,k].set_ylabel('Amount',fontsize=fontsize)
    axs[1,k].set_ylim([0, 1])
    PlotStyle(axs[1,k])

        
    ynormB0 = (solB[:,0]-solB[:,0].min())/(solB[:,0].max()-solB[:,0].min())
    ynormB1 = (solB[:,1]-solB[:,1].min())/(solB[:,1].max()-solB[:,1].min())
    
    if ynormB0.max()!=1 or ynormB1.max()!=1:
        ynormB0 = np.ones(1000)
        ynormB1 = np.ones(1000)
    
    axs[2,k].plot(t,ynormB0,color='red')
    axs[2,k].plot(t,ynormB1,color='black')
    axs[2,k].text(7.5,0.85,'Z0',fontsize=14,color='red')
    axs[2,k].text(8.5,0.85,'>>>',fontsize=14,alpha=0.5)
    axs[2,k].text(10.25,0.85,'Z1',fontsize=14,color='black')
    axs[2,k].set_xlabel('Time (days)',fontsize=fontsize)
    axs[2,k].set_ylabel('Amount',fontsize=fontsize)
    axs[2,k].set_ylim([0, 1])
    PlotStyle(axs[2,k])

letters = ['A','B','C']
for jj in range(3):
    axs[jj,0].text(0.01, 0.99, letters[jj], size=25, color='black', ha='left', va='top', transform=axs[jj,0].transAxes)

plt.tight_layout()
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr04/data/images/image_dyn_fragments.png',dpi=75,bbox_inches='tight')
plt.close()