#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 02:46:44 2023

@author: tavo
"""

import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.colors import Normalize

###############################################################################
# Loading packages 
###############################################################################
  
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

data = pd.read_csv('/media/tavo/storage/sunspots/selecteddata.csv')
data['date']  = pd.to_datetime(data['date'])
data['SPdelta'] = data['SurfPres_Forecast_A'] - data['SurfPres_Forecast_D']
data['SSTdelta'] = data['SurfSkinTemp_A'] - data['SurfSkinTemp_D']
data['SATdelta'] = data['SurfAirTemp_A'] - data['SurfAirTemp_D']
data['CFdelta'] = data['CloudFrc_A'] - data['CloudFrc_D']
data['T03delta'] = data['TotO3_A'] - data['TotO3_D']
data['OLRdelta'] = data['OLR_A'] - data['OLR_D']
data['COLRdelta'] = data['ClrOLR_A'] - data['ClrOLR_D']

dataspots = pd.read_csv('/media/tavo/storage/sunspots/sunspots.csv')
dataspots['date'] = pd.to_datetime(dataspots['date'])
rollingavgspots = dataspots.groupby('date')['dailysunspots'].mean()

wlnames = ['UVA','UVB','UVC','Vis','NIR','SWIR']
wdata = pd.read_csv('/media/tavo/storage/sunspots/solarinterpolated.csv')
wdata['date'] = pd.to_datetime(wdata['date'])
wdata = wdata.set_index('date')
wdata = wdata.rolling(7).apply(np.median)


for val in wlnames:    
    data[val] = np.array(wdata[val].loc[data['date']])

groupdata = data.groupby(pd.qcut(data['frac_scale_norm'],365,duplicates='drop')).mean().rolling(14).mean()

columns = ['SPdelta','SSTdelta','SATdelta','CFdelta','T03delta','OLRdelta',
           'COLRdelta','UVA','UVB','UVC','Vis','NIR','SWIR']

correlation = groupdata[columns].corr().values

###############################################################################
# Loading packages 
###############################################################################

cmap = cm.get_cmap('viridis')
normalizer = Normalize(groupdata['cases'].min(),groupdata['cases'].max())
im = cm.ScalarMappable(norm=normalizer)

nvars = len(columns)

fig = plt.figure(figsize=(20,20))
gs = gridspec.GridSpec(nrows=nvars+1, ncols=nvars) 
axs = np.array([fig.add_subplot(gs[k,j]) for k in range(nvars) for j in range(nvars)]).reshape((nvars,nvars))

for k,val in enumerate(columns):
    for j,sal in enumerate(columns):
        if k!=j:
            axs[k,j].scatter(groupdata[val].values,groupdata[sal].values,
                             c=groupdata['cases'].values,alpha=np.abs(correlation[k,j]))        
        ImageStyle(axs[k,j])
        
    axs[k,0].set_ylabel(val,fontsize=16)
    axs[-1,k].set_xlabel(val,fontsize=16)

axmp = fig.add_subplot(gs[-1,:])
cbaxes = axmp.inset_axes([0,0.5,1,0.15])
cbar = plt.colorbar(im,cax=cbaxes, orientation='horizontal')
cbar.set_label('Mean Number of Cases',fontsize=16)

ImageStyle(axmp)

plt.tight_layout()
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr2/images/image_cases_env.png',dpi=75,bbox_inches='tight')
plt.close()

