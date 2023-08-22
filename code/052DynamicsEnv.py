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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import interpolate
from itertools import product
from matplotlib.colors import Normalize

###############################################################################
# Loading packages 
###############################################################################

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 5
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])

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

def MakeShapePlot(Rolling,labelA,labelB,shapeColumns,ax,scale=0.05):
    
    xp = Rolling[labelA]
    yp = Rolling[labelB]
    shapeData = Rolling[shapeColumns]
    colordata = Rolling.index
    scale = 0.05
    
    if type(colordata[0])!=np.int64:
        colordata = np.array([val.left for val in colordata])
        
    colordata = (colordata - colordata.min())/(colordata.max() - colordata.min())
    colors = [plt.cm.viridis(each) for each in colordata]
    
    ax.scatter(xp,yp,s=1,color='black')
    
    for k,(xx,yy) in enumerate(zip(xp,yp)):
        
        y = list(np.sort(np.array(shapeData.iloc[k])))
        x = np.linspace(0,1,num=int(len(y)/2)+1)
        x = list(x[1::][::-1])+list(x[1::])
    
        tck, u = interpolate.splprep([x + x[:1], y + y[:1]], s=0, per=True)
        unew = np.linspace(0, 1, 100)
        
        basic_form = interpolate.splev(unew, tck)
        xm,ym = np.mean(basic_form[0]),np.mean(basic_form[1])
        
        xcoords,ycoords = scale*(basic_form[0]-xm)+xx, scale*(basic_form[1]-ym)+yy
        ax.plot(xcoords, ycoords, color=colors[k], lw=1)
        ax.fill(xcoords, ycoords, color=colors[k], alpha=0.25)

###############################################################################
# Loading packages 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
MetaData = MetaData.set_index('id')

KmerData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/sequences/KmerDataUpd2023.csv')
KmerData = KmerData.set_index('id')
KmerData = KmerData.loc[MetaData.index]

data = pd.concat([MetaData,KmerData],axis=1)

data['date'] = pd.to_datetime(data['date'])

data['SPdelta'] = data['SurfPres_Forecast_A'] - data['SurfPres_Forecast_D']
data['SSTdelta'] = data['SurfSkinTemp_A'] - data['SurfSkinTemp_D']
data['SATdelta'] = data['SurfAirTemp_A'] - data['SurfAirTemp_D']
data['CFdelta'] = data['CloudFrc_A'] - data['CloudFrc_D']
data['T03delta'] = data['TotO3_A'] - data['TotO3_D']
data['OLRdelta'] = data['OLR_A'] - data['OLR_D']
data['COLRdelta'] = data['ClrOLR_A'] - data['ClrOLR_D']

###############################################################################
# Loading packages 
###############################################################################

def ProcessSeries(rolling):
    
    rolling = rolling.mean()
    rolling = rolling.dropna()
    rolling = (rolling-rolling.min())/(rolling.max()-rolling.min())
    
    return rolling

yearCalendar = data.groupby('dayofyear').mean(numeric_only=True).rolling(15)
yearSolar = data.groupby(pd.qcut(data['daylength'],365,duplicates='drop')).mean(numeric_only=True).rolling(15)
solar = data.groupby('sunspots').mean(numeric_only=True).rolling(15)

yearCalendar = ProcessSeries(yearCalendar)
yearSolar = ProcessSeries(yearSolar)
solar = ProcessSeries(solar)

names = ['DOY \n CFdelta','SD \n CFdelta','Spots \n CFdelta']
namesx = ['T03delta \n 1-mer','T03delta \n 2-mer','T03delta \n 3-mer','T03delta \n 4-mer']

anames = ['DOY \n UVB','SD \n UVB','Spots \n UVB']
anamesx = ['UVC \n 1-mer','UVC \n 2-mer','UVC \n 3-mer','UVC \n 4-mer']

namescbar = ['DOY','SD','Spots']
ltrs = ['A','B','C']

fig = plt.figure(figsize=(20,25))
gs = gridspec.GridSpec(nrows=9, ncols=4) 

for k,val in enumerate([yearCalendar,yearSolar,solar]):
    
    axs = [fig.add_subplot(gs[3*k,j]) for j in range(len(Blocks))]
    axs[0].text(0.01, 0.99, ltrs[k], size=25, color='black', ha='left', va='top', transform=axs[0].transAxes)
    
    for j,sal in enumerate(Blocks):
        MakeShapePlot(val,'T03delta','CFdelta',Blocks[j],axs[j],scale=0.1)
        ImageStyle(axs[j])
        axs[j].set_xlabel(namesx[j],fontsize=16)        
    axs[0].set_ylabel(names[k],fontsize=16)
    
    axs = [fig.add_subplot(gs[3*k+1,j]) for j in range(len(Blocks))]
    
    for j,sal in enumerate(Blocks):
        MakeShapePlot(val,'UVB','UVC',Blocks[j],axs[j],scale=0.1)
        ImageStyle(axs[j])
        axs[j].set_xlabel(anamesx[j],fontsize=16)        
    axs[0].set_ylabel(anames[k],fontsize=16)
    
    cmap = cm.get_cmap('viridis')
    if type(val.index[0])!=np.int64:
        minVal = val.index.min().left
        maxVal = val.index.max().left
    else:
        minVal = val.index.min()
        maxVal = val.index.max()        
    normalizer = Normalize(minVal,maxVal)
    
    
    localax = fig.add_subplot(gs[3*k+2,:])
    cbaxes = localax.inset_axes([0,0.5,1,0.1])
    
    im = cm.ScalarMappable(norm=normalizer)
    cbar = plt.colorbar(im,cax=cbaxes, orientation='horizontal')
    cbar.set_label('Mean '+namescbar[k],fontsize=16)
    ImageStyle(localax)

plt.tight_layout()
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr04/data/images/image_dynenv.png',dpi=75,bbox_inches='tight')
plt.close()