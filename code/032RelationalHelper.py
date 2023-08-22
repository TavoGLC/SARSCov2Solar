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
import matplotlib.pyplot as plt

from joblib import dump
from sklearn import preprocessing as pr

from typing import Sequence,Tuple

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

sh = 10**-6

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * sh * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def MainLoss(Model,params,batchStats,z_rng ,batch,batch_target,weight):
  
    block, newbatchst = Model().apply({'params': params, 'batch_stats': batchStats}, batch, z_rng,mutable=['batch_stats'])
    recon_x, mean, logvar, hlp = block
    kld_loss = weight*kl_divergence(mean, logvar).mean()
    loss_value = optax.l2_loss(recon_x, batch).mean()
    help_loss = optax.l2_loss(hlp, batch_target).mean()
    total_loss = loss_value + kld_loss + 0.1*help_loss
    
    return total_loss,newbatchst['batch_stats']

def TrainModel(TrainDataSet,TestDataSet,Loss,params,batchStats,rng,epochs=10,batch_size=64,lr=0.005):
    
    TrainData,TrainTargets = TrainDataSet
    TestData, TestTargets = TestDataSet
    
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
    def step(params,batchStats ,optState, z_rng, batch,batch_targets,weight):
        
        (loss_value,batchStats), grads = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch,batch_targets,weight)
        updates, optState = localOptimizer.update(grads, optState, params)
        params = optax.apply_updates(params, updates)
        
        return params,batchStats, optState, loss_value
    
    @jax.jit
    def getloss(params,batchStats, z_rng, batch,batch_targets):
        (loss_value,_), _ = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch,batch_targets,1)
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
            b_targets = TrainTargets[k:k+batch_size]
        
            rng, key = random.split(rng)
            params,batchStats ,optState, lossval = step(params,batchStats,optState,key,batch,b_targets,aWeights[ii])
            losses.append(lossval)
            batchtime.append(time.time()-stb)
            ii = ii+1
        
        valloss = []
        for i in range(0,len(TestData),batch_size):            
            rng, key = random.split(rng)
            val_batchPaths = TestData[i:i+batch_size]
            val_batch = LoadBatch(val_batchPaths)
            val_targets = TestTargets[i:i+batch_size]
            valloss.append(getloss(params,batchStats,key,val_batch,val_targets))
        
        mbatch = 1000*np.mean(batchtime)
        meanloss = np.mean(losses)
        meanvalloss = np.mean(valloss)
        
        trainloss.append(meanloss)
        testloss.append(meanvalloss)
        
        localIndex = np.arange(len(TrainData))
        np.random.shuffle(localIndex)
        
        np.random.shuffle(TrainData)
        
        TrainData = TrainData[localIndex]
        TrainTargets = TrainTargets[localIndex]
    
        end = time.time()
        output = 'Epoch = '+str(epoch) + ' Time per epoch = ' + str(round(end-st,3)) + 's  Time per batch = ' + str(round(mbatch,3)) + 'ms' + ' Train Loss = ' + str(meanloss) +' Test Loss = ' + str(meanvalloss)
        print(output)
        
    return trainloss,testloss,params,batchStats

##############################################################################
# Data loading 
###############################################################################

foldsPath = r'/media/tavo/storage/biologicalSequences/covidsr04/data/folds/BiasedA'

TrainFolds = pd.read_csv(foldsPath+'/train.csv')
TestFolds = pd.read_csv(foldsPath+'/test.csv')
Validation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/test.csv')

foldNames = ['Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4']

basePath = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/relational'

outputPath = '/media/tavo/storage/biologicalSequences/covidsr04/data/Models/RelationalSupervised'

Validation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/test.csv')

###############################################################################
# Loading packages 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')

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

batchSize = 64
InputShape = (16,16,16,1)

depth = 2
mainUnits = [4,12,12,4]

auxColumn = ['sunspots','UVB','Vis','NIR','SPdelta','SSTdelta','SATdelta','CFdelta','T03delta']

###############################################################################
# Loading packages 
###############################################################################

modelcontainer = []
losscont = []
vallosscont = []

for k,nme in enumerate(foldNames):
    
    np.random.seed(128)
    trainIndex = TrainFolds[nme]
    testIndex = TestFolds[nme]
    
    #trainLabels = trainIndex[0:batchSize*100]
    #testLabels = testIndex[0:batchSize*100]
    
    trainLabels = trainIndex[0:batchSize*(trainIndex.shape[0]//batchSize)]
    testLabels = testIndex[0:batchSize*(testIndex.shape[0]//batchSize)]
    
    trainData = np.array([basePath+'/'+val+'.npy' for val in trainLabels])
    testData = np.array([basePath+'/'+val+'.npy' for val in testLabels])
    
    trainAux = MetaData[auxColumn].loc[trainLabels].values
    testAux = MetaData[auxColumn].loc[testLabels].values
    
    scalerh = pr.QuantileTransformer()
    scalerh.fit(trainAux)
    dump(scalerh,outputPath + '/scalerh'+nme+'.joblib')
    
    trainAux = scalerh.transform(trainAux)
    testAux = scalerh.transform(testAux)
    
    
    def VAEModel():
        return ConvHelperVAE(mainUnits,(3,3,3),(2,2,2),InputShape,1,depth,batchSize,len(auxColumn))
     
    def loss(params,batchStats,z_rng ,batch,batch_target,weigth):
        return MainLoss(VAEModel,params,batchStats,z_rng ,batch,batch_target,weigth)
     
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
     
    finalShape = tuple([batchSize]+list(InputShape))
    init_data = jnp.ones(finalShape, jnp.float32)
    initModel = VAEModel().init(key, init_data, rng)
    
    params0 = initModel['params']
    batchStats = initModel['batch_stats']

    trloss,tstloss,params0,batchStats = TrainModel([trainData,trainAux],[testData,testAux],loss,params0,
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

valData = np.array([basePath+'/'+val+'.npy' for val in Validation['validation']])#[0:100*batchSize]

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
    
    localparams = {'params':prs['params']['encoder'],'batch_stats':prs['batch_stats']['encoder']}

    def EncoderModel(batch):
        return CONVEncoder(mainUnits,(3,3,3),(2,2,2),InputShape,depth,batchSize,train=False).apply(localparams, batch)
    
    VariationalRep = TransformData(EncoderModel,valData,1*batchSize)
    axs[ki].scatter(VariationalRep[:,0],VariationalRep[:,1],alpha=0.05)
    axs[ki].title.set_text('Latent Space (model = ' + str(k) +')')
    PlotStyle(axs[ki])
    
plt.tight_layout()
plt.savefig(outputPath+'/figls.png')
plt.close()
