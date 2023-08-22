#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:42:34 2023

@author: tavo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.stats

from itertools import product

###############################################################################
# Visualization functions
###############################################################################
fontsize = 16

def BottomStyle(Axes): 
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
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=fontsize)
    Axes.yaxis.set_ticks([])
    
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

Alphabet = ['A','C','T','G']
Blocks = []

maxSize = 5
for k in range(1,maxSize):
    
    Blocks.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
n_rolling = 30
q_rolling = 100

###############################################################################
# Loading packages 
###############################################################################

data = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv')
data['date'] = pd.to_datetime(data['date'])

###############################################################################
# Loading packages 
###############################################################################

def ProcessData(rollingdf):
    
    rollingmean = rollingdf.mean()
    rollingsem = rollingdf.sem()
    rollingerror = scipy.stats.t.ppf(1.95/2,n_rolling-1) * rollingsem
    
    return rollingmean,rollingerror

def MakeRollingPlot(rollingdf,ax):
    
    rmean,rerr = ProcessData(rollingdf)
    
    if type(rmean.index[0])==pd._libs.interval.Interval:    
        xvals = [val.right for val in rmean.index[0:-1]]
    else:
        xvals = rmean.index[0:-1]
    
    ax.plot(xvals,rmean.values[0:-1],color='navy')
    ax.plot(xvals,(rmean-rerr).values[0:-1],color='black',alpha=0.5)
    ax.plot(xvals,(rmean+rerr).values[0:-1],color='black',alpha=0.5)
    
    ax.fill_between(xvals, (rmean-rerr).values[0:-1], (rmean+rerr).values[0:-1],color='black', alpha=0.125)

r0 = data.groupby('date')['effectivelength'].mean().rolling(14)
r1 = data.groupby(pd.qcut(data['daylength'],365,duplicates='drop'))['effectivelength'].mean().rolling(14)
r2 = data.groupby('sunspots')['effectivelength'].mean().rolling(14)

###############################################################################
# Loading packages 
###############################################################################

cities = data['query'].value_counts().index

city_loc = cities[0]
sdata = data[data['query']==city_loc].copy()

daily_rolling = 7*2

###############################################################################
# Loading packages 
###############################################################################

fig = plt.figure(figsize=(20,25))
gs = gridspec.GridSpec(nrows=7, ncols=3) 

ax = fig.add_subplot(gs[0,:]) 
MakeRollingPlot(r0,ax=ax)
ax.set_ylabel("Mean Genome of\n Size",fontsize=fontsize)
ax.set_xlabel("Date",fontsize=fontsize)
ax.text(0.01, 0.99, 'A', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)

ax = fig.add_subplot(gs[1,:]) 
MakeRollingPlot(r1,ax=ax)
ax.set_ylabel("Mean Genome of\n Size",fontsize=fontsize)
ax.set_xlabel("SD",fontsize=fontsize)
ax.text(0.01, 0.99, 'B', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)

ax = fig.add_subplot(gs[2,:]) 
MakeRollingPlot(r2,ax=ax)
ax.set_ylabel("Mean Genome of\n Size",fontsize=fontsize)
ax.set_xlabel("Sun Spots",fontsize=fontsize)
ax.text(0.01, 0.99, 'C', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)

ax = fig.add_subplot(gs[3::,0]) 

days = np.arange(1,365)

for k,val in enumerate(days):
    miniData = sdata[sdata['dayofyear']==val]
    a,b = np.histogram(miniData['effectivelength'],bins=100)
    b = [(b[j]+b[j+1])/2 for j in range(len(b)-1)]
    a = (a - a.min())/(a.max() - a.min())
    ax.plot(b,4*a+k,color='black',alpha=0.25)

ax.set_xlim(29500,29775)
ax.set_ylim(0,370)
ax.set_xlabel("Mean Effective Genome Size",fontsize=fontsize)
ax.set_ylabel("DOY",fontsize=fontsize)
ax.text(0.01, 0.99, 'D', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)
    
ax = fig.add_subplot(gs[3::,1]) 

ldays = sdata['daylength'].unique()

for k,val in enumerate(ldays):
    miniData = sdata[sdata['daylength']==val]
    a,b = np.histogram(miniData['effectivelength'],bins=100)
    b = [(b[j]+b[j+1])/2 for j in range(len(b)-1)]
    a = (a - a.min())/(a.max() - a.min())
    ax.plot(b,0.075*a+val,color='black',alpha=0.25)

ax.set_xlim(29500,29775)
ax.set_ylim(ldays.min(),ldays.max()+.1)
ax.set_xlabel("Mean Effective Genome Size",fontsize=fontsize)
ax.set_ylabel("SD",fontsize=fontsize)
ax.text(0.01, 0.99, 'E', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)
    
###############################################################################
# Loading packages 
###############################################################################

ax = fig.add_subplot(gs[3,2]) 
data.groupby('dayofyear')['effectivelength'].mean().diff().plot(ax=ax,color='navy')
ax.set_xlabel("DOY",fontsize=fontsize)
ax.set_ylabel("Difference in Mean \n Sequence Size",fontsize=fontsize)
ax.text(0.01, 0.99, 'F', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)
    
ax = fig.add_subplot(gs[4,2]) 
rolling_a = data.groupby('dayofyear')['effectivelength'].mean().diff().rolling(daily_rolling).std()
rolling_a.plot(ax=ax,color='navy')
ax.set_xlabel("DOY",fontsize=fontsize)
ax.set_ylabel("Sequence \n  Size Volatility",fontsize=fontsize)
ax.text(0.01, 0.99, 'G', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)
    
ax = fig.add_subplot(gs[5,2]) 
rolling_b = data.groupby('dayofyear')['Pangolin'].value_counts().unstack().T.mean().rolling(daily_rolling).mean()
rolling_b.plot(ax=ax,color='navy')
ax.set_xlabel("DOY",fontsize=fontsize)
ax.set_ylabel("Mean Number of \n  Isolated Variants",fontsize=fontsize)
ax.text(0.01, 0.99, 'H', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)
    
ax = fig.add_subplot(gs[6,2]) 
ax.scatter(rolling_a.values,rolling_b.values,color='navy')
ax.set_xlabel("Sequence Size Volatility",fontsize=fontsize)
ax.set_ylabel("Mean Number of\n Isolated Variants",fontsize=fontsize)
ax.text(0.01, 0.99, 'I', size=25, color='black', ha='left', va='top', transform=ax.transAxes)
txt = 'Correlation = ' + str(round(np.corrcoef(rolling_a.values[daily_rolling::],rolling_b.values[daily_rolling::])[0,1],2))
ax.text(0.65, 0.99, txt, size=15, color='black', ha='left', va='top', transform=ax.transAxes)
PlotStyle(ax)

plt.tight_layout()
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr04/data/images/image_daily.png',dpi=75,bbox_inches='tight')
plt.close()

