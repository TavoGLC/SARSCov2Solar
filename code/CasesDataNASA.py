#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:50:42 2023

@author: tavo
"""

import os
import numpy as np
import pandas as pd 
import multiprocessing as mp
import matplotlib.pyplot as plt
import differint.differint as df
import matplotlib.gridspec as gridspec

from PIL import Image
from scipy.signal import find_peaks
from mpl_toolkits.basemap import Basemap

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.interpolate import RectSphereBivariateSpline

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

def GetDayLenght(J,lat):
    #CERES model  Ecological Modelling 80 (1995) 87-95
    phi = 0.4093*np.sin(0.0172*(J-82.2))
    coef = (-np.sin(np.pi*lat/180)*np.sin(phi)-0.1047)/(np.cos(np.pi*lat/180)*np.cos(phi))
    ha =7.639*np.arccos(np.max([-0.87,coef]))
    return ha

def GetYearLengths(lat):
    return np.array([GetDayLenght(j, lat) for j in range(368)])

def GetDictsBylat(lat,function):
    
    Dict = {}
    Dict_n = {}
    
    days = function(lat)
    days_n = (days - days.min())/(days.max() - days.min())
    
    inDict = {}
    inDict_n = {}
    
    for k,_ in enumerate(days):
        inDict[k] = days[k]
        inDict_n[k] = days_n[k]
    
    Dict[0] = inDict
    Dict_n[0] = inDict_n
    
    for j in range(1,4):
        
        localdf = df.GL(j/3,days,num_points=len(days))
        localdf_n = (localdf - localdf.min())/(localdf.max() - localdf.min())
        
        inDict = {}
        inDict_n = {}
        
        for i,_ in enumerate(localdf):
            inDict[i] = localdf[i]
            inDict_n[i] = localdf_n[i]
        
        Dict[j] = inDict
        Dict_n[j] = inDict_n
    
    return Dict,Dict_n

MaxCPUCount=int(0.85*mp.cpu_count())

def GetDictsBylatDL(lat):
    return GetDictsBylat(lat,GetYearLengths)

#Wraper function for parallelization 
def GetDataParallel(data,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    mData=localPool.map(Function, [ val for val in data])
    localPool.close()
    
    return mData

###############################################################################
# Loading packages 
###############################################################################

dataspots = pd.read_csv('/media/tavo/storage/sunspots/sunspots.csv')
dataspots['date'] = pd.to_datetime(dataspots['date'])
rollingavgspots = dataspots.groupby('date')['dailysunspots'].mean()

wldata = pd.read_csv('/media/tavo/storage/sunspots/solarcurrent.csv')
wldata = wldata[wldata['irradiance']>0]

wbins = [200,290,320,400,700,1000,2500]
wlnames = ['UVA','UVB','UVC','Vis','NIR','SWIR']

sdata = wldata.groupby(['date',pd.cut(wldata['wavelength'],wbins)])['irradiance'].mean().unstack()
sdata.columns = wlnames

sdata = sdata.reset_index()
sdata['date'] = pd.to_datetime(sdata['date'])
sdata = sdata.set_index('date')

for val in wlnames:
    
    mean = sdata[val].mean()
    std = sdata[val].std()
    
    sdata[val] = [sal if np.abs((sal-mean)/std)<4 else mean for sal in sdata[val]]

###############################################################################
# Loading packages 
###############################################################################

cases_columns = ['date','cases','country','qry','lat','long','dayofyear','year','lengthofday']

dataam = pd.read_csv('/media/tavo/storage/sunspots/continental2.csv')
dataeu = pd.read_csv('/media/tavo/storage/sunspots/europe.csv')

dataam = dataam[cases_columns]
dataeu = dataeu[cases_columns]

data = pd.concat([dataam,dataeu],axis=0)

data['date'] = pd.to_datetime(data['date'])
data['spots'] = np.array(rollingavgspots.loc[data['date']])
data['refdate'] = ['2019-12-01 00:00:00' for val in data['date']]
data['refdate'] = pd.to_datetime(data['refdate'])
data['days_since_initialcase'] = (data['date'] - data['refdate'] ).dt.days

data['normspots'] = data['spots']/data['lengthofday']

###############################################################################
# Loading packages 
###############################################################################

mainindex = sdata.index.intersection(data['date'])

correctqrys = [val for val in data['qry'].unique() if val !='lat==0.0 & long==0.0']
newdata = data[data['qry'].isin(correctqrys)]
counts = newdata['qry'].value_counts()
highcounts = counts[counts>700].index

newdata = newdata[newdata['qry'].isin(highcounts)]
newdata = newdata[newdata['date'].isin(mainindex)].copy()

finalqrys = newdata['qry'].unique()

def MakeFeatures(dataset):
    
    for val in wlnames:
        dataset[val] = np.array(sdata[val].loc[dataset['date']])

    for val in wlnames:
        dataset['norm'+val] = dataset[val]/dataset['lengthofday']
        
    return dataset

newdata = MakeFeatures(newdata)

###############################################################################
# Loading packages 
###############################################################################

shortrolling_n = 7*2

container = []

for k,val in enumerate(finalqrys):
    
    selected = newdata[newdata['qry']==val].copy()      
    cLat = selected['lat'].mean()
    cQry = selected['qry'].iloc[0]
    
    shortrolling = selected.groupby('dayofyear')['cases'].apply(np.median)
    shortrolling = shortrolling.to_frame()
    
    lengths = [GetDayLenght(J, cLat) for J in range(0,366)]
    disc_range = [k for k in range(172,192)]
    forder = 0
    
    for i in range(5,20):
        
        corder = i/20
        diff_dl = df.GL(corder,lengths,num_points=len(lengths))
        shortrolling['fractional'] = diff_dl[shortrolling.index-1]
        shortrolling['norm_fractional'] = (shortrolling['fractional']-shortrolling['fractional'].min())/(shortrolling['fractional'].max()-shortrolling['fractional'].min())
        short_disc = shortrolling.groupby('norm_fractional')['cases'].mean().rolling(shortrolling_n).mean().argmax()
        
        if short_disc in disc_range:
            forder = corder
            break
    
    diff_dl = df.GL(forder,lengths,num_points=len(lengths))
    dayToDiff = {}
    for k,val in enumerate(diff_dl):
        dayToDiff[k+1] = val
        
    selected['frac_scale'] = [dayToDiff[val] for val in selected['dayofyear']]
    selected['frac_scale_norm'] = (selected['frac_scale'] - selected['frac_scale'].min())/(selected['frac_scale'].max() - selected['frac_scale'].min())
    selected['frac_value'] = [forder for val in selected['dayofyear']]
    
    container.append(selected)

datascale = pd.concat(container)

###############################################################################
# Loading packages 
###############################################################################

dataorg = '/media/tavo/storage/nasa_npy/'
datadir = dataorg + 'data/'

files = np.sort(os.listdir(datadir))
filePaths = [datadir+val for val in files]

uniqueDates = datascale['date'].unique()

###############################################################################
# Loading packages 
###############################################################################

lats = np.linspace(0.1, 179, 180) * np.pi / 180
lons = np.linspace(0.1, 359, 360) * np.pi / 180

toConcat = []

names = ['SurfPres_Forecast_A','SurfSkinTemp_A','SurfAirTemp_A','CloudFrc_A',
         'TotO3_A','OLR_A','ClrOLR_A','SurfPres_Forecast_D','SurfSkinTemp_D',
         'SurfAirTemp_D','CloudFrc_D','TotO3_D','OLR_D','ClrOLR_D']

for dte in uniqueDates:
    
    cop = datascale[datascale['date']==dte].copy()
    
    lat = ((cop['lat']+90)*np.pi/180).values
    long = ((cop['long']+180)*np.pi/180).values
    targetfile = 'rollingmean_70_'+str(dte)+'.npy'
    
    if targetfile in files:
        
        currentPath = datadir+targetfile
    else:
        currentPath = filePaths[0]
        
    for kk in range(len(names)):
        data = np.load(currentPath)
        interpolator = RectSphereBivariateSpline(lats, lons, data[kk])
        results = []
        for lt,ln in zip(lat,long):
            results.append(interpolator(lt,ln)[0][0])
        cop[names[kk]] = results
        
    toConcat.append(cop)

toConcat = pd.concat(toConcat)

toConcat.to_csv('/media/tavo/storage/sunspots/selecteddata.csv',index=False)
