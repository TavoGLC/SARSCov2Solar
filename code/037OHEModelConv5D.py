#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License

Copyright (c) 2023 Octavio Gonzalez-Lugo 

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

from typing import Sequence,Tuple
from scipy.sparse import coo_array

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
# modified from https://github.com/pvjosue/pytorch_convNd/blob/master/convNd.py
class Conv4D(nn.Module):

    Channels: Sequence[int]
    Ksize: Tuple[int]
    Name: str
    
    Strides: Tuple[int] = (1,1,1,1)
    
    def setup(self):
        
        self.padding = 'SAME'
        l_k, d_k, h_k, w_k = self.Ksize
        self.conv3dlist = [nn.Conv(self.Channels,(d_k, h_k, w_k),use_bias=False,padding='SAME',name=self.Name+'conv4D'+str(k)) for k in range(l_k)]    
        
    def __call__(self,inputs):
        
        inputs_shape = inputs.shape
        _, l_i, _, _, _, _ = inputs_shape
        l_k,l_s = self.Ksize[0], self.Strides[0]
        p_before, p_after = 0, 0
        l_o = l_i
        
        # output tensors for each 3D frame
        frame_results = [None] * l_o
        at_least_one_output = False
        
        for i_kernel in range(l_k):  # kernel
            for j_input in range(l_i):  # input
               
                kernel_offset = (l_k // 2) - (1 - l_k % 2)
                out_frame_padded = j_input - i_kernel + p_before + kernel_offset
                out_frame_idx = out_frame_padded - (l_k - 1) // 2

                if out_frame_idx % l_s:
                    continue
                out_frame_tot = out_frame_idx
                out_frame_idx = out_frame_idx // l_s
                if out_frame_idx < 0 or out_frame_idx >= l_o:
                    continue
                
                channel_last_inputs = inputs[:, j_input, ...]
                
                frame_conv3d = self.conv3dlist[i_kernel](channel_last_inputs)
                at_least_one_output = True
                outconv_shape = frame_conv3d.shape
                if frame_results[out_frame_idx] is None:
                    frame_results[out_frame_idx] = frame_conv3d
                else:
                    frame_results[out_frame_idx] += frame_conv3d
        
        # With insufficient padding, all frames could become border frames could be empty
        if not at_least_one_output:
            raise ValueError('The combination of parameters used in 4D cNNs has yield an empty tensor')
        # With excessive padding, some frames are zero
        for i, frame in enumerate(frame_results):
            if frame is None:
                frame_results[i] = jnp.zeros(shape=outconv_shape)

        output = jnp.stack(frame_results, axis=1)

        return output

class Conv5D(nn.Module):

    Channels: Sequence[int]
    Ksize: Tuple[int]
    Name: str
    
    Strides: Tuple[int] = (1,1,1,1,1)
    
    def setup(self):
        
        self.padding = 'SAME'
        l_k, s_k, d_k, h_k, w_k = self.Ksize
        self.conv5dlist = [Conv4D(self.Channels,(s_k, d_k, h_k, w_k),self.Name+'conv5D'+str(k)) for k in range(l_k)]    
        
    def __call__(self,inputs):
        
        inputs_shape = inputs.shape
        _, l_i, _, _, _, _, _ = inputs_shape
        l_k,l_s = self.Ksize[0], self.Strides[0]
        p_before, p_after = 0, 0
        l_o = l_i
        
        # output tensors for each 3D frame
        frame_results = [None] * l_o
        at_least_one_output = False
        
        for i_kernel in range(l_k):  # kernel
            for j_input in range(l_i):  # input
               
                kernel_offset = (l_k // 2) - (1 - l_k % 2)
                out_frame_padded = j_input - i_kernel + p_before + kernel_offset
                out_frame_idx = out_frame_padded - (l_k - 1) // 2

                if out_frame_idx % l_s:
                    continue
                out_frame_tot = out_frame_idx
                out_frame_idx = out_frame_idx // l_s
                if out_frame_idx < 0 or out_frame_idx >= l_o:
                    continue
                
                channel_last_inputs = inputs[:, j_input, ...]
                
                frame_conv5d = self.conv5dlist[i_kernel](channel_last_inputs)
                at_least_one_output = True
                outconv_shape = frame_conv5d.shape
                if frame_results[out_frame_idx] is None:
                    frame_results[out_frame_idx] = frame_conv5d
                else:
                    frame_results[out_frame_idx] += frame_conv5d
        
        # With insufficient padding, all frames could become border frames could be empty
        if not at_least_one_output:
            raise ValueError('The combination of parameters used in 4D cNNs has yield an empty tensor')
        # With excessive padding, some frames are zero
        for i, frame in enumerate(frame_results):
            if frame is None:
                frame_results[i] = jnp.zeros(shape=outconv_shape)

        output = jnp.stack(frame_results, axis=1)

        return output

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
        finalDepth = [self.depth-k if self.depth-k>=1 else 1 for k,val in enumerate(self.Units)]
        
        if self.UpSampling:
            finalDepth = finalDepth[::-1]
        
        for k,val in enumerate(self.Units):
            
            for ii in range(finalDepth[k]):
                if len(self.Ksize)==4:
                    x = Conv4D(val,self.Ksize,self.Name+' conv_conv_'+str(k)+str(ii))(x)
                elif len(self.Ksize)==5:
                    x = Conv5D(val,self.Ksize,self.Name+' conv_conv_'+str(k)+str(ii))(x)
                else:    
                    x = nn.Conv(val,self.Ksize,padding='SAME',use_bias=False,
                                name=self.Name+' conv_conv_'+str(k)+str(ii))(x)
                x = nn.BatchNorm(use_running_average=not self.train,
                                 name = self.Name+' conv_norm_'+str(k)+str(ii))(x)
                x = nn.leaky_relu(x)
                
            if self.UpSampling:
                
                localShape = x.shape
                toChange = [val*sal for val,sal in zip(localShape[1:-1],self.Strides)]
                upShape = [localShape[0]]+toChange+[localShape[-1]]
                x = jax.image.resize(x, tuple(upShape), 'nearest')
                
            else:
                x = nn.avg_pool(x,self.Ksize,self.Strides,padding='SAME')
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
        self.encoder = CoderMLP(self.Units[1::],'encodermlp',train=self.train)
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

class EncoderNetwork(nn.Module):
    
    train: bool = True 
    
    def setup(self):
        
        self.localConv5D = CoderCONV([48],(3,3,3,3,3),(2,2,2,2,2),2,
                                   'conv5Dencoder',UpSampling=False,
                                   train=self.train)
        
        self.localConv4D = CoderCONV([64],(3,3,3,3),(2,2,2,2),2,
                                   'conv4Dencoder',UpSampling=False,
                                    train=self.train)
                
        self.localConv3D = CoderCONV([64],(3,3,3),(2,2,2),2,
                                   'conv3Dencoder',UpSampling=False,
                                   train=self.train)
        
        self.localConv2D = CoderCONV([64,16],(3,3),(2,2),2,
                                   'conv2Dencoder',UpSampling=False,
                                   train=self.train)
        
        self.localEncoder = EncoderMLP([64,8,2],
                                       train=self.train)
        
    def __call__(self,inputs):
        
        x = inputs
        x = self.localConv5D(x)
        x = x.reshape((x.shape[0],8,8,8,8,12))
        x = self.localConv4D(x)
        x = x.reshape((x.shape[0],8,8,8,32))
        x = self.localConv3D(x)
        x = x.reshape((x.shape[0],8,8,64))
        x = self.localConv2D(x)
        x = x.reshape((x.shape[0],-1))
        mean_x,logvar_x = self.localEncoder(x)
        
        return mean_x,logvar_x

class DecoderNetwork(nn.Module):
    
    train: bool = True 
    
    def setup(self):
        
        self.localConv5D = CoderCONV([24],(3,3,3,3,3),(2,2,2,2,2),2,
                                      'conv5Ddecoder',UpSampling=True,
                                      train=self.train)
        
        self.localConv4D = CoderCONV([32],(3,3,3,3),(2,2,2,2),2,
                                      'conv4Ddecoder',UpSampling=True,
                                      train=self.train)
        
        self.localConv3D = CoderCONV([32],(3,3,3),(2,2,2),2,
                                   'conv3Ddecoder',UpSampling=True,
                                   train=self.train)
        
        self.localConv2D = CoderCONV([16,16],(3,3),(2,2),2,
                                   'conv2Ddecoder',UpSampling=True,
                                   train=self.train)
        
        self.localDecoder = DecoderMLP([2,8,64],
                                       train=self.train)
        self.outnorm = nn.BatchNorm(use_running_average=not self.train,name = 'outnorm')
        self.outConv = Conv5D(4,(3,3,3,3,3),'conv_dec_out')
        
    #@nn.compact
    def __call__(self,inputs):
        
        x = inputs
        x = self.localDecoder(x)
        x = x.reshape((x.shape[0],2,2,16))
        x = self.localConv2D(x)
        x = x.reshape((x.shape[0],4,4,4,16))
        x = self.localConv3D(x)
        x = x.reshape((x.shape[0],4,4,4,4,64))
        x = self.localConv4D(x)
        x = x.reshape((x.shape[0],4,4,4,4,4,128))
        x = self.localConv5D(x)
        x = self.outConv(x)
        x = self.outnorm(x)
        x = nn.softmax(x)
        
        return x

###############################################################################
# Loading packages 
###############################################################################

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std

class ConvVAE(nn.Module):
    
    train: bool = True 
    
    def setup(self):
        self.encoder = EncoderNetwork(self.train)
        self.decoder = DecoderNetwork(self.train)
        
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
        encoded = np.load(pth)
        container.append(encoded)
    container = np.stack(container)
    
    return container.astype(np.float32) 

def DataLoader(datadirs,batchsize):
    for k in range(0,len(datadirs),batchsize):
        paths = datadirs[k:k+batchsize]
        yield LoadBatch(paths)
        
###############################################################################
# Loading packages 
###############################################################################

sh = 10**-4

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * sh * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def MainLoss(Model,params,batchStats,z_rng ,batch):
  
    block, newbatchst = Model().apply({'params': params, 'batch_stats': batchStats}, batch, z_rng,mutable=['batch_stats'])
    recon_x, mean, logvar = block
    
    kld_loss = kl_divergence(mean, logvar).mean()
    loss_recon = optax.sigmoid_binary_cross_entropy(recon_x, batch).mean()
    
    total_loss = loss_recon + kld_loss
    
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
    
    localOptimizer = optax.chain(optax.clip_by_global_norm(1),
                                 optax.scale_by_adam(),  
                                 optax.scale_by_schedule(Scheduler), 
                                 optax.scale(-1.0)
                                 )
    
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
        
        for batch in DataLoader(TrainData,batch_size):

            stb = time.time()
        
            rng, key = random.split(rng)
            params,batchStats ,optState, lossval = step(params,batchStats,optState,key,batch)
            losses.append(lossval)
            batchtime.append(time.time()-stb)

        
        valloss = []
        for val_batch in DataLoader(TestData,batch_size):
            
            rng, key = random.split(rng)
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

foldsPath = r'/media/tavo/storage/biologicalSequences/covidsr04/data/folds/clustered'

TrainFolds = pd.read_csv(foldsPath+'/train.csv')
TestFolds = pd.read_csv(foldsPath+'/test.csv')
Validation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/folds/test.csv')

foldNames = ['Fold0', 'Fold1', 'Fold2', 'Fold3', 'Fold4']
#foldNames = ['Fold4']

basePath = '/media/tavo/storage/biologicalSequences/covidsr04/data/OHEE/'

outputPath = '/media/tavo/storage/biologicalSequences/covidsr04/data/Models/OHEConv5D'

###############################################################################
# Loading packages 
###############################################################################

batchSize = 16
InputShape = (8,8,8,8,8,4)

x0 = jnp.ones((16,8,8,8,8,8,4))
x1 = jnp.ones((16,2))
rng = random.PRNGKey(0)
#print(EncoderNetwork().tabulate(jax.random.key(0), x0,console_kwargs={'width':150}))
#print(DecoderNetwork().tabulate(jax.random.key(0), x1,console_kwargs={'width':150}))

print(ConvVAE().tabulate(jax.random.key(0), x0,rng,console_kwargs={'width':150}))

###############################################################################
# Loading packages 
###############################################################################
modelcontainer = []
losscont = []
vallosscont = []

for k,nme in enumerate(foldNames):
    
    np.random.seed(128)
    trainLabels = TrainFolds[nme].values
    testLabels = TestFolds[nme].values
    
    trainSamps = np.array([basePath+'/'+val+'.npy' for val in trainLabels])
    testSamps = np.array([basePath+'/'+val+'.npy' for val in testLabels])

    trainData = trainSamps[0:batchSize*(trainSamps.shape[0]//batchSize)]
    testData = testSamps[0:batchSize*(testSamps.shape[0]//batchSize)]
    
    def VAEModel():
        return ConvVAE()
     
    def loss(params,batchStats,z_rng ,batch):
        return MainLoss(VAEModel,params,batchStats,z_rng ,batch)
     
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
     
    finalShape = tuple([batchSize]+list(InputShape))
    init_data = jnp.ones(finalShape, jnp.float32)
    initModel = VAEModel().init(key, init_data, rng)
    
    params0 = initModel['params']
    batchStats = initModel['batch_stats']

    trloss,tstloss,params0,batchStats = TrainModel(trainData,testData,loss,params0,
                                                   batchStats,rng,lr=0.0025,
                                                   epochs=4,batch_size=batchSize)
    
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

valData = np.array([basePath+'/'+val+'.npy' for val in Validation['validation']])

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

for k,prs in enumerate(modelcontainer):
    
    localparams = {'params':prs['params']['encoder'],'batch_stats':prs['batch_stats']['encoder']}

    def EncoderModel(batch):
        return EncoderNetwork(train=False).apply(localparams, batch)
    
    VariationalRep = TransformData(EncoderModel,valData,batchSize)
    axs[k].scatter(VariationalRep[:,0],VariationalRep[:,1],alpha=0.05)
    axs[k].title.set_text('Latent Space (model = ' + str(k) +')')
    PlotStyle(axs[k])
    
plt.tight_layout()  
plt.savefig(outputPath+'/figls.png')
plt.close() 
