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

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from typing import Sequence,Tuple, Callable

import jax
import optax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from flax.serialization import to_bytes

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
    
###############################################################################
# Loading packages 
###############################################################################
class CoderCONV(nn.Module):
    
    Units: Sequence[int]
    Ksize: Tuple[int]
    Strides: Tuple[int]
    
    depth: int
    name: str 
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
                            name=self.name+' conv_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.name+' conv_norm_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
                
            if self.UpSampling:
                x = nn.ConvTranspose(val,self.Ksize,padding='SAME',
                                     strides=self.Strides,use_bias=False,
                                     name=self.name+' convUp_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.name+' conv_normUp_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
            else:
                x = nn.Conv(val,self.Ksize,padding='SAME',strides=self.Strides,
                            use_bias=False,
                            name=self.name+' convDn_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.name+' conv_normDn_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
            
        return x

class CoderMLP(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,feat in enumerate(self.Units):
            x = nn.Dense(feat,use_bias=False,name = self.name+' layer_'+str(k))(x)
            x = nn.BatchNorm(use_running_average=not self.train,name = self.name+' norm_'+str(k))(x)
            x = nn.leaky_relu(x)
        return x

###############################################################################
# Loading packages 
###############################################################################

class EncoderMLP(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.encoder = CoderMLP(self.Units[1::],self.name,train=self.train)
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
    name: str 
    train: bool = True
    
    def setup(self):
        self.decoder = CoderMLP(self.Units[0:-1],self.name,train=self.train)
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
    name: str 
    train: bool = True 
    
    def setup(self):
        
        self.localConv = CoderCONV(self.Units,self.Ksize,self.Strides,self.depth,self.name+'conv_enc',UpSampling=False,train=self.train)
        self.divFactor = 2**(len(self.Units)-1)
        
        self.targetShape = [val//self.divFactor for val in self.InputShape[0:-1]] + [self.Units[-1]]
        
        self.localShape = np.prod(np.array(self.targetShape))
        self.EncUnits = [self.localShape,self.localShape//4,self.localShape//16,2]
        self.localEncoder = EncoderMLP(self.EncUnits,self.name+'mlp_enc',train=self.train)
        
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
    name: str 
    train: bool = True 
    
    def setup(self):
        
        self.localConv = CoderCONV(self.Units[1::],self.Ksize,self.Strides,self.depth,self.name+'conv_dec',UpSampling=True,train=self.train)
        self.divFactor = 2**(len(self.Units)-1)
        
        self.finalShape = [self.BatchSize]+[val//self.divFactor for val in self.InputShape[0:-1]] + [self.Units[-1]]
        
        self.localShape = np.prod(np.array(self.finalShape[1::]))
        self.DecUnits = [2,self.localShape//16,self.localShape//4,self.localShape]
        
        self.localDecoder = DecoderMLP(self.DecUnits,self.name+'mlp_dec',train=self.train)
        
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
                                   self.name+'main_enc',self.train)
        self.decoder = CONVDecoder(self.Units[::-1],self.Ksize,self.Strides,
                                   self.InputShape,self.outchannels,
                                   self.depth,self.BatchSize,
                                   self.name+'main_dec',self.train)

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
    
    return container.reshape((-1,16,16,16,2))

def DataLoader(datadirs,batchsize):
    for k in range(0,len(datadirs),batchsize):
        paths = datadirs[k:k+batchsize]
        yield LoadBatch(paths)
        
###############################################################################
# Loading packages 
###############################################################################

sh = 10**-6

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * sh * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def MainLoss(Model,params,batchStats,z_rng ,batch,weight):
  
    block, newbatchst = Model().apply({'params': params, 'batch_stats': batchStats}, batch, z_rng,mutable=['batch_stats'])
    recon_x, mean, logvar = block
    kld_loss = weight*kl_divergence(mean, logvar).mean()
    loss_value = optax.l2_loss(recon_x, batch).mean()
    total_loss = loss_value + kld_loss
    
    return total_loss,newbatchst['batch_stats']

def TrainModel(TrainData,TestData,Loss,params,batchStats,rng,epochs=10,batch_size=64,lr=0.005):
    
    totalSteps = epochs*(TrainData.shape[0]//batch_size) + epochs
    stepsPerCycle = totalSteps//4

    esp = [{"init_value":lr/10, 
            "peak_value":(lr)/((k+1)), 
            "decay_steps":int(stepsPerCycle*0.75), 
            "warmup_steps":int(stepsPerCycle*0.25), 
            "end_value":lr/10} for k in range(4)]
    
    Scheduler = optax.sgdr_schedule(esp)

    localOptimizer = optax.adam(learning_rate=Scheduler)
    optState = localOptimizer.init(params)
    
    aWeights = MakeAnnealingWeights(totalSteps,4)
    
    @jax.jit
    def step(params,batchStats ,optState, z_rng, batch,weight):
        
        (loss_value,batchStats), grads = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch,weight)
        updates, optState = localOptimizer.update(grads, optState, params)
        params = optax.apply_updates(params, updates)
        
        return params,batchStats, optState, loss_value
    
    @jax.jit
    def getloss(params,batchStats, z_rng, batch):
        (loss_value,_), _ = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch,1)
        return loss_value
    
    trainloss = []
    testloss = []
    ii = 0
    for epoch in range(epochs):
        
        st = time.time()
        batchtime = []
        losses = []
        
        for k in range(0,len(TrainData),batch_size):
    
            stb = time.time()
            batchPaths = TrainData[k:k+batch_size]
            batch = LoadBatch(batchPaths)
        
            rng, key = random.split(rng)
            params,batchStats ,optState, lossval = step(params,batchStats,optState,key,batch,aWeights[ii])
            losses.append(lossval)
            batchtime.append(time.time()-stb)
            ii = ii+1
        
        valloss = []
        for i in range(0,len(TestData),batch_size):            
            rng, key = random.split(rng)
            val_batchPaths = TestData[i:i+batch_size]
            val_batch = LoadBatch(val_batchPaths)
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

##############################################################################
# Data loading 
###############################################################################

foldsPath = r'/media/tavo/storage/biologicalSequences/covidsr2/Folds'

TrainFolds = pd.read_csv(foldsPath+'/train.csv')
TestFolds = pd.read_csv(foldsPath+'/test.csv')
Validation = pd.read_csv(foldsPath+'/validation.csv')

foldNames = ['Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4']

basePath = '/media/tavo/storage/biologicalSequences/covid/datasets/Relational'

outputPath = '/media/tavo/storage/biologicalSequences/covidsr2/Models/Relational'

###############################################################################
# Loading packages 
###############################################################################

batchSize = 64
InputShape = (16,16,16,2)

depth = 2
mainUnits = [4,12,12,4]

###############################################################################
# Loading packages 
###############################################################################

modelcontainer = []
losscont = []
vallosscont = []

for k,nme in enumerate(foldNames):
    
    np.random.seed(128)
    trainLabels = TrainFolds[nme]
    testLabels = TestFolds[nme]
    
    trainSamps = np.array([basePath+'/'+val+'.npy' for val in trainLabels])
    testSamps = np.array([basePath+'/'+val+'.npy' for val in testLabels])
    
    trainData = trainSamps[0:batchSize*(trainSamps.shape[0]//batchSize)]
    testData = testSamps[0:batchSize*(testSamps.shape[0]//batchSize)]
    
    def VAEModel():
        return ConvVAE(mainUnits,(3,3,3),(2,2,2),InputShape,2,depth,batchSize,'test')
     
    def loss(params,batchStats,z_rng ,batch,weigth):
        return MainLoss(VAEModel,params,batchStats,z_rng ,batch,weigth)
     
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
     
    finalShape = tuple([batchSize]+list(InputShape))
    init_data = jnp.ones(finalShape, jnp.float32)
    initModel = VAEModel().init(key, init_data, rng)
    
    params0 = initModel['params']
    batchStats = initModel['batch_stats']

    trloss,tstloss,params0,batchStats = TrainModel(trainData,testData,loss,params0,
                                                   batchStats,rng,lr=0.0025,
                                                   epochs=15,batch_size=batchSize)
    
    finalParams = {'params':params0,'batch_stats':batchStats}
    modelcontainer.append(finalParams)
    losscont.append(trloss)
    vallosscont.append(tstloss)
    
    model_file = outputPath + '/flax_model'+str(k)
    
    with open(model_file, "wb") as f:
        model_bytes = to_bytes(finalParams)
        f.write(model_bytes)

###############################################################################
# Learning curves
###############################################################################

fig,axs = plt.subplots(1,5,figsize=(30,10))

for kk,block in enumerate(zip(losscont,vallosscont)):
    val,sal = block
    axs[kk].plot(val,'k-',label = 'Loss')
    axs[kk].plot(sal,'r-',label = 'Validation Loss')
    axs[kk].title.set_text('Reconstruction loss')
    axs[kk].legend()
    PlotStyle(axs[kk])
    
plt.tight_layout()
plt.savefig(outputPath+'/figtraining.png')
plt.close()

valData = np.array([basePath+'/'+val+'.npy' for val in Validation['val']])[0:100000]

###############################################################################
# Learning curves
###############################################################################

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

fig,axs = plt.subplots(1,5,figsize=(30,10))

for ki,prs in enumerate(modelcontainer):
    
    localparams = {'params':prs['params']['testmain_enc'],'batch_stats':prs['batch_stats']['testmain_enc']}

    def EncoderModel(batch):
        return CONVEncoder(mainUnits,(3,3,3),(2,2,2),InputShape,depth,batchSize,'testmain_enc',train=False).apply(localparams, batch)
    
    VariationalRep = TransformData(EncoderModel,valData,1*batchSize)
    axs[ki].scatter(VariationalRep[:,0],VariationalRep[:,1],alpha=0.05)
    axs[ki].title.set_text('Latent Space (model = ' + str(k) +')')
    PlotStyle(axs[ki])
    
plt.tight_layout()
plt.savefig(outputPath+'/figls.png')
plt.close() 