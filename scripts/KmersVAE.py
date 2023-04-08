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

from Bio import SeqIO
from io import StringIO
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
    name: str 
    train: bool = True 
    
    def setup(self):
        self.layers = [nn.Dense(feat,use_bias=False,name = self.name+' layer_'+str(k)) for k,feat in enumerate(self.Units)]
        self.norms = [nn.BatchNorm(use_running_average=not self.train,name = self.name+' norm_'+str(k)) for k,feat in enumerate(self.Units)]
        
    @nn.compact
    def __call__(self,inputs):
        x = inputs
        for k,block in enumerate(zip(self.layers,self.norms)):
            lay,norm = block
            x = lay(x)
            x = norm(x)
            x = nn.relu(x)
        return x

class Encoder(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.encoder = Coder(self.Units[1::],self.name,self.train)
        self.mean = nn.Dense(self.Units[-1], name='mean')
        self.logvar = nn.Dense(self.Units[-1], name='logvar')
    
    @nn.compact
    def __call__(self, inputs):
        
        x = inputs
        mlpencoded = self.encoder(x)
        mean_x = self.mean(mlpencoded)
        logvar_x = self.logvar(mlpencoded)
        
        return mean_x, logvar_x

class Decoder(nn.Module):
    
    Units: Sequence[int]
    name: str 
    train: bool = True
    
    def setup(self):
        self.decoder = Coder(self.Units[0:-1],self.name,self.train)
        self.out = nn.Dense(self.Units[-1],use_bias=False, name='out')
    
    @nn.compact
    def __call__(self, inputs):
        x = inputs
        decoded_1 = self.decoder(x)
        
        out =self.out(decoded_1)
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
    name: str 
    train: bool = True
    
    def setup(self):
        self.encoder = Encoder(self.Units,self.name+'encoder',self.train)
        self.decoder = Decoder(self.Units[::-1],self.name+'decoder',self.train)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

mainUnits  = [340,170,85,21,5,2]

def VAEModel():
    return VAE(mainUnits,'test')

###############################################################################
# Loading packages 
###############################################################################

Alphabet = ['A','C','T','G']
Labels = []

maxSize = 5
for k in range(1,maxSize):
    
    Labels.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
KmerLabels = [item for sublist in Labels for item in sublist]

def GetSeq(path):

    with open(path) as file:
        
        seqData=file.read()
        
    Seq=StringIO(seqData)
    SeqList=list(SeqIO.parse(Seq,'fasta'))
    
    return SeqList[0]

###############################################################################
# Data selection
###############################################################################

DataPath = r"/media/tavo/storage/biologicalSequences/covid/datasets"
DataDir = DataPath+'/KmerDataUpdSmall.csv'
KmerData = pd.read_csv(DataDir)
KmerData = KmerData.set_index('id')

###############################################################################
# Data selection
###############################################################################

headers = ['id'] + KmerLabels 

KmerData = pd.read_csv(DataDir,usecols=headers)
KmerData = KmerData.set_index('id')

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/MetaData.csv')
MetaData = MetaData[MetaData['correctdata']==1]
MetaData = MetaData.set_index('id')

KmerData = KmerData.loc[MetaData.index]

###############################################################################
# Loading packages 
###############################################################################

basePath = '/media/tavo/storage/biologicalSequences/covid/models/Model0jax'

Models = ['/flax_model'+str(k) for k in range(5)]
Scalers = ['/scalerFold'+str(k)+'.joblib' for k in range(5)]

index = 0

with open(basePath+Scalers[index], "rb") as scaler_file:
    Scaler = joblib.load(scaler_file)

with open(basePath+Models[index], "rb") as state_f:
    state = from_bytes(VAEModel, state_f.read())
    state = jax.tree_util.tree_map(jnp.array, state)

rng = random.PRNGKey(0)
KmerDataTR = Scaler.transform(np.array(KmerData))
localparams = {'params':state['params']['testencoder'],'batch_stats':state['batch_stats']['testencoder']}

def EcoderModel(trainparams,batch):
    return Encoder(mainUnits,'testencoder',train=False).apply(trainparams, batch)

VariationalRep = []
Bsize = 80000
for k in range(0,len(KmerDataTR),Bsize):
    localfrag = KmerDataTR[k:k+Bsize]
    mu,logvar = EcoderModel(localparams,localfrag)
    varfrag = reparameterize(rng,mu,logvar)
    VariationalRep.append(varfrag)

VariationalRep = np.vstack(VariationalRep)

del KmerDataTR

if index==1:    
    MetaData['ldim0'] = VariationalRep[:,1]
    MetaData['ldim1'] = VariationalRep[:,0]
else:
    MetaData['ldim0'] = VariationalRep[:,0]
    MetaData['ldim1'] = VariationalRep[:,1]


###############################################################################
# Loading packages 
###############################################################################

order = np.argsort(VariationalRep[:,1])
ordval = np.array([val for val in order if np.abs(VariationalRep[val,1])<0.0025])
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

dataspots = pd.read_csv('/media/tavo/storage/sunspots/sunspots.csv')
dataspots['date'] = pd.to_datetime(dataspots['date'])
rollingavgspots = dataspots.groupby('date')['dailysunspots'].mean()

MiniData['spots'] = np.array(rollingavgspots.loc[MiniData['date']])
MiniData['normspots'] = MiniData['spots']/MiniData['lengthofday']

wldata = pd.read_csv('/home/tavo/Documentos/solarcurrent.csv')
wldata = wldata[wldata['irradiance']>0]

wbins = [200,290,320,400,700,1000,2500]
wlnames = ['UVA','UVB','UVC','Vis','NIR','SWIR']

data = wldata.groupby(['date',pd.cut(wldata['wavelength'],wbins)])['irradiance'].mean().unstack()
data.columns = wlnames

data = data.reset_index()
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

data = data.rolling(20).apply(np.median)

intersectIndex = data.index.intersection(MiniData['date'])

MiniData = MiniData[MiniData['date'].isin(intersectIndex)]

for val in wlnames:    
    MiniData[val] = np.array(data[val].loc[MiniData['date']])
    
for val in wlnames:
    MiniData['norm'+val] = MiniData[val]/MiniData['lengthofday']
    
for val in wlnames:
    MiniData['diff'+val] = np.array((data[val].diff()).loc[MiniData['date']])
    
for val in wlnames:
    MiniData['normdiff'+val] = MiniData['diff'+val]/MiniData['lengthofdayd10']
    
###############################################################################
# Loading packages 
###############################################################################

featlistg =['lengthofday','lengthofdayd10','ldim0','ldim1',
            'dayofyear','normspots','UVA','UVB','UVC','Vis',
            'NIR','SWIR']

plotnames =['lengthofday','lengthofdayd10','dayofyear','normspots',
            'UVA','UVB','UVC','Vis','NIR','SWIR']

correlation  = MiniData[featlistg].corr()

###############################################################################
# Loading packages 
###############################################################################

def MakeDimReductionPlot(Data,column,colname,ax):
    localPlot = ax.scatter(Data['ldim0'],Data['ldim1'],c=Data[column],alpha=0.2,cmap='Blues')
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
MakeDimReductionPlot(MetaData,'lengthofdayd10','Sunshine Duration Rate of Change (SDRC)',axs2)
PlotStyle(axs2)
axs2.hlines(y=0.0025,xmin=MetaData['ldim0'].min(),xmax=MetaData['ldim0'].max(),linewidth=1, color='navy',alpha=0.5)
axs2.hlines(y=-0.0025,xmin=MetaData['ldim0'].min(),xmax=MetaData['ldim0'].max(),linewidth=1, color='navy',alpha=0.5)
axs2.text(0.01, 0.99, 'C', size=25, color='black', ha='left', va='top', transform=axs2.transAxes)

###############################################################################
# Loading packages 
###############################################################################

axs = [fig.add_subplot(gs[k,3:6]) for k in range(6)]
xlabs = ['SD','SDRC','DOY','NS','UVA','UVB','UVC','Vis','NIR','SWIR']
correlation['ldim0'].abs()[plotnames].plot.bar(color='navy',ax=axs[0])
axs[0].set_ylabel('Absolute Pearson Correlation')
axs[0].set_xticks(np.arange(len(plotnames)),xlabs,rotation=45)
axs[0].text(0.01, 0.99, 'D', size=25, color='black', ha='left', va='top', transform=axs[0].transAxes)
PlotStyle(axs[0])

###############################################################################
# Loading packages 
###############################################################################

n_rolling = 100
features = ['dayofyear','lengthofdayd10','normspots','Vis','Length'] 
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
plt.savefig('/media/tavo/storage/images/image_fold_'+str(index)+'.png',dpi=75,bbox_inches='tight')
plt.close()
