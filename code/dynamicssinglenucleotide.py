#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 00:34:09 2023

@author: tavo
"""

###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.patches import Ellipse

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

def MakePanelCByBlock(DataFrame,column,block,cax):
    
    localData = DataFrame
    a,b,c,d = block
    
    cax.scatter(np.array(localData[a]),np.array(localData[b]),
                color=[0,0,0,1],s=1,alpha=0.5)
    
    if column=='date':
        dcolors = np.linspace(0,1,num=len(localData.index))
    elif column.find('lengthofday')==0:
        inData = [val.right for val in localData.index]
        dcolors = np.array(inData)
        dcolors = (dcolors-dcolors.min())/(dcolors.max()-dcolors.min())
    else:
        dcolors = np.array(localData.index)
        dcolors = (dcolors-dcolors.min())/(dcolors.max()-dcolors.min())

    colors = [plt.cm.viridis(each) for each in dcolors]
    
    for x,y,xer,yer,col in zip(localData[a],localData[b],localData[c],localData[d],colors):
        el = cax.add_artist(Ellipse((x,y),xer/10,yer/10))
        el.set_clip_box(cax.bbox)
        el.set_alpha(0.75)
        el.set_facecolor(col[0:3])
    
    PlotStyle(cax)


###############################################################################
# Loading packages 
###############################################################################

Alphabet = ['A','C','T','G']
n_rolling = 30
q_rolling = 100

colrs = [plt.cm.viridis(val) for val in np.linspace(0,1,num=len(Alphabet))]

###############################################################################
# Loading packages 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/MetaDataNASA.csv')
MetaData = MetaData[MetaData['correctdata']==1]
MetaData = MetaData[MetaData['qry']!='lat==0.0 & long==0.0']

MetaData = MetaData.set_index('id')

KmerData = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/KmerDataUpd.csv',usecols=['id']+Alphabet)
KmerData = KmerData.set_index('id')
KmerData = KmerData.loc[MetaData.index]

data = pd.concat([MetaData,KmerData],axis=1)

data['date'] = pd.to_datetime(data['date'])

dataspots = pd.read_csv('/media/tavo/storage/sunspots/sunspots.csv')
dataspots['date'] = pd.to_datetime(dataspots['date'])
rollingavgspots = dataspots.groupby('date')['dailysunspots'].mean()

data['spots'] = np.array(rollingavgspots.loc[data['date']])
data['normspots'] = data['spots']/data['lengthofday']
data['normspots_sf'] = data['spots']/data['sf_msd']


###############################################################################
# Loading packages 
###############################################################################

fig = plt.figure(figsize=(25,25))
gs = gridspec.GridSpec(nrows=6, ncols=24) 

labels = ['date','dayofyear','spots']
names = ['Date','DOY','Sunspots']
letters = ['A','B','C','D','E','F','G','H','I','J']
li = 0

for k,lab in enumerate(labels):
    
    axs0 = fig.add_subplot(gs[k,0:15])

    localRolling = data.groupby(lab)[Alphabet].mean().rolling(n_rolling)
    rollingMean = localRolling.mean()
    rollingMean = (rollingMean-rollingMean.min())/(rollingMean.max()-rollingMean.min())
    rollingMean.plot(ax=axs0,color=colrs)
    axs0.legend(loc=1)
    PlotStyle(axs0)
    axs0.set_xlabel(names[k],fontsize=fontsize)
    axs0.set_ylabel('Normalized content',fontsize=fontsize)
    axs0.text(0.01, 0.99, letters[li], size=25, color='black', ha='left', va='top', transform=axs0.transAxes)
    li = li+1
    
    if lab == 'date':
        axs0.vlines(['2020-02-6','2020-03-12'],ymin=0,ymax=1,color='black') #Remdesivir in adults with severe COVID-19: a randomised, double-blind, placebo-controlled, multicentre trial
        axs0.vlines(['2020-02-21','2020-04-19'],ymin=0,ymax=1,color='gray') #Remdesivir for the Treatment of Covid-19 — Final Report 
        axs0.vlines(['2020-03-22','2020-10-04'],ymin=0,ymax=1,color='red') #Repurposed Antiviral Drugs for Covid-19 — Interim WHO Solidarity Trial Results

    axs1 = fig.add_subplot(gs[k,15:-1])
    cp = MakePanelCByBlock(rollingMean.dropna(),lab,Alphabet,axs1)
    axs1.text(0.01, 0.99, letters[li], size=25, color='black', ha='left', va='top', transform=axs1.transAxes)
    li = li+1
    axs1cb = fig.add_subplot(gs[k,-1])
    
    if lab=='date':
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=rollingMean.index.min().value,
                                                  vmax=rollingMean.index.max().value))
        cbar = fig.colorbar(sm,cax=axs1cb)
        tks = pd.to_datetime(cbar.get_ticks()).strftime(date_format='%b %Y')
        cbar.ax.set_yticks(cbar.get_ticks().tolist())
        cbar.ax.set_yticklabels(tks)
        cbar.ax.set_ylabel(names[k], rotation=270)
    else:
        cbar = fig.colorbar(cp,cax=axs1cb,cmap='viridis')
        cbar.ax.set_ylabel(names[k], rotation=270)

labels = ['lengthofday']
names = ['SD','0.6SD','SDRC']

for k,lab in enumerate(labels):
    
    axs6 = fig.add_subplot(gs[k+3,0:15])

    localRolling = data[Alphabet].groupby(pd.qcut(data[lab],1000,duplicates='drop')).mean().rolling(q_rolling)
    rollingMean = localRolling.mean()
    rollingMean = (rollingMean-rollingMean.min())/(rollingMean.max()-rollingMean.min())
    rollingMean.plot(ax=axs6,color=colrs)
    axs6.legend(loc=1)
    axs6.xaxis.set_tick_params(rotation=10)
    PlotStyle(axs6)
    axs6.set_xlabel(names[k],fontsize=fontsize)
    axs6.set_ylabel('Normalized content',fontsize=fontsize)
    axs6.text(0.01, 0.99, letters[li], size=25, color='black', ha='left', va='top', transform=axs6.transAxes)
    li = li+1
    
    axs7 = fig.add_subplot(gs[k+3,15:-1])
    cp = MakePanelCByBlock(rollingMean.dropna(),lab,Alphabet,axs7)
    axs7.text(0.01, 0.99, letters[li], size=25, color='black', ha='left', va='top', transform=axs7.transAxes)
    li = li+1
    
    axs7cb = fig.add_subplot(gs[k+3,-1])
    cbar = fig.colorbar(cp,cax=axs7cb,cmap='viridis')
    cbar.ax.set_ylabel(names[k], rotation=270)

plt.tight_layout()
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr2/images/image_dyn.png',dpi=75,bbox_inches='tight')
plt.close()