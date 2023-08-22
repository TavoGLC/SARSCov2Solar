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

from typing import Sequence, Callable

import jax
import optax
from jax import lax 
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

#FullData
foldsPath = r'/media/tavo/storage/biologicalSequences/covidsr04/data/folds/Full'

TrainFolds = pd.read_csv(foldsPath+'/train.csv')
TestFolds = pd.read_csv(foldsPath+'/test.csv')
Validation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/test.csv')

DataDir = '/media/tavo/storage/biologicalSequences/covidsr04/sequences/KmerDataUpdSmall2023.csv'

#FullData
outputPath = r'/media/tavo/storage/biologicalSequences/covidsr04/data/Models/KmersSupervised'

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
# Data selection
###############################################################################

KmerData = pd.read_csv(DataDir)
KmerData = KmerData.set_index('id')

###############################################################################
# Data selection
###############################################################################

ScalersContainer = []
ScalersHContainer = []
foldNames = ['Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4']

for fold in foldNames:
    
    trainLabels = TrainFolds[fold]
    trainData = np.array(KmerData.loc[trainLabels])    
    trainAux = np.array(MetaData[auxColumn].loc[trainLabels])    
    
    scaler = pr.MinMaxScaler()
    scaler.fit(trainData)
    ScalersContainer.append(scaler)
    dump(scaler,outputPath + '/scaler'+fold+'.joblib')
    
    scalerh = pr.QuantileTransformer()
    scalerh.fit(trainAux)
    ScalersHContainer.append(scalerh)
    dump(scalerh,outputPath + '/scalerh'+fold+'.joblib')
    
###############################################################################
# Loading packages 
###############################################################################

def MainLossHelper(Model,params,batchStats,z_rng ,batch,batch_target):
  
    block, newbatchst = Model().apply({'params': params, 'batch_stats': batchStats}, batch, z_rng,mutable=['batch_stats'])
    recon_x, mean, logvar, z, hlp,ret = block

    regloss = 0
    for w in jax.tree_util.tree_leaves(params):
        regloss = regloss + lambda_reg*jnp.linalg.norm(w,ord=1)
    
    kld_loss = kl_divergence(mean, logvar).mean(axis=-1)
    loss_value = optax.l2_loss(recon_x, batch).mean(axis=-1)
    help_loss = optax.l2_loss(hlp, batch_target).mean(axis=-1)
    dyn_loss = dynamics(z,ret).mean()
    total_loss = loss_value.mean() + kld_loss.mean() + 0.01*help_loss.mean() + dyn_loss +regloss
    
    return total_loss,newbatchst['batch_stats']

def TrainModelHelper(TrainDataSet,TestDataSet,Loss,params,batchStats,rng,epochs=10,batch_size=64,lr=0.005):
    
    TrainData,TrainTargets = TrainDataSet
    TestData, TestTargets = TestDataSet
    
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
    def step(params,batchStats ,optState, z_rng, batch,batch_targets):
        
        (loss_value,batchStats), grads = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch,batch_targets)
        updates, optState = localOptimizer.update(grads, optState, params)
        params = optax.apply_updates(params, updates)
        
        return params,batchStats, optState, loss_value
    
    @jax.jit
    def getloss(params,batchStats, z_rng, batch,batch_targets):
        (loss_value,_), _ = jax.value_and_grad(Loss,has_aux=True)(params,batchStats, z_rng, batch,batch_targets)
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
            b_targets = TrainTargets[k:k+batch_size]
            
            rng, key = random.split(rng)
            params,batchStats ,optState, lossval = step(params,batchStats,optState,key,batch,b_targets)
            losses.append(lossval)
            batchtime.append(time.time()-stb)
        
        valloss = []
        for i in range(0,TestData.shape[0],batch_size):
            rng, key = random.split(rng)
            val_batch = TestData[i:i+batch_size]
            val_targets = TestTargets[i:i+batch_size]
            valloss.append(getloss(params,batchStats,key,val_batch,val_targets))
        
        mbatch = 1000*np.mean(batchtime)
        meanloss = np.mean(losses)
        meanvalloss = np.mean(valloss)
        
        trainloss.append(meanloss)
        testloss.append(meanvalloss)
        localIndex = np.arange(len(TrainData))
        np.random.shuffle(localIndex)
        
        TrainData = TrainData[localIndex,:]
        TrainTargets = TrainTargets[localIndex]
    
        end = time.time()
        output = 'Epoch = '+str(epoch) + ' Time per epoch = ' + str(round(end-st,3)) + 's  Time per batch = ' + str(round(mbatch,3)) + 'ms' + ' Train Loss = ' + str(meanloss) +' Test Loss = ' + str(meanvalloss)
        print(output)
        
    return trainloss,testloss,params,batchStats

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

    trainData = np.array(KmerData.loc[trainLabels])
    testData = np.array(KmerData.loc[testLabels])
    
    trainAuxData = np.array(MetaData[auxColumn].loc[trainLabels])
    testAuxData = np.array(MetaData[auxColumn].loc[testLabels])
    
    trainData = ScalersContainer[k].transform(trainData)
    testData = ScalersContainer[k].transform(testData)  
    
    trainAuxData = ScalersHContainer[k].transform(trainAuxData)
    testAuxData = ScalersHContainer[k].transform(testAuxData)  
    
    def VAEHelperModel():
        return VAEHelper(mainUnits,hUnits,len(auxColumn))
    
    def loss(params,batchStats,z_rng ,batch,batch_targets):
        return MainLossHelper(VAEHelperModel,params,batchStats,z_rng ,batch,batch_targets)

    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    
    init_data = jnp.ones((batchSize, InputShape), jnp.float32)
    initModelH = VAEHelperModel().init(key, init_data, rng)
    
    params0H = initModelH['params']
    batchStatsH = initModelH['batch_stats']

    trloss,tstloss,params0H,batchStatsH = TrainModelHelper([trainData,trainAuxData],[testData,testAuxData],loss,params0H,batchStatsH,rng,lr=0.001,epochs=50,batch_size=batchSize)
    finalParamsH = {'params':params0H,'batch_stats':batchStatsH}

    model_file = outputPath+ '/flax_model'+str(k)
    
    with open(model_file, "wb") as f:
        model_bytes = to_bytes(finalParamsH)
        f.write(model_bytes)
    
    modelcontainer.append(finalParamsH)
    losscont.append(trloss)
    vallosscont.append(tstloss)

###############################################################################
# Learning curves
###############################################################################

RepresentationContainer = []

fig,axs = plt.subplots(1,5,figsize=(30,10))

for k,block in enumerate(zip(losscont,vallosscont)):
    val,sal = block
    axs[k].plot(val,'k-',label = 'Loss')
    axs[k].plot(sal,'r-',label = 'Validation Loss')
    axs[k].title.set_text('Reconstruction loss')
    axs[k].legend()
    PlotStyle(axs[k])
    
plt.tight_layout()
plt.savefig(outputPath+'/figtraining.png')
plt.close()

###############################################################################
# Learning curves
###############################################################################

#Full data
validationData = np.array(KmerData.loc[Validation['validation']])

performance = []

fig,axs = plt.subplots(1,5,figsize=(30,10))

for k,block in enumerate(zip(ScalersContainer,modelcontainer)):
    
    sclr,trainparamas = block
    
    valData = sclr.transform(validationData)
    
    def EcoderModel(trainparams,batch):
        return Encoder(mainUnits,train=False).apply(trainparams, batch)
    
    localparams = {'params':trainparamas['params']['encoder'],'batch_stats':trainparamas['batch_stats']['encoder']}
    
    mu,logvar = EcoderModel(localparams,valData)
    VariationalRepresentation = reparameterize(rng,mu,logvar)
    
    axs[k].scatter(VariationalRepresentation[:,0],VariationalRepresentation[:,1],alpha=0.15)
    axs[k].title.set_text('Latent Space (model = ' + str(k) +')')
    PlotStyle(axs[k])
    
plt.tight_layout()  
plt.savefig(outputPath+'/figls.png')
plt.close() 

