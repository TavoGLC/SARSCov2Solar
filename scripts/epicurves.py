#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:30:10 2023

@author: tavo
"""

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

weatherdata = pd.read_csv('/media/tavo/storage/sunspots/weatherdata.csv')
weatherdata['date'] = pd.to_datetime(weatherdata['date'])

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/MetaData.csv')

wldata = pd.read_csv('/home/tavo/Documentos/solarcurrent.csv')
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

lats = data['lat'].unique()

localdicts = GetDataParallel(lats,GetDictsBylatDL)

qryToDL= {}
qryToDL_n = {}

for val,sal in zip(lats,localdicts):
    qryToDL[val] = sal[0]
    qryToDL_n[val] = sal[1]

data['lengthofdayd03'] = [qryToDL[val][1][sal] for val,sal in zip(data['lat'],data['dayofyear'])]
data['lengthofdayd06'] = [qryToDL[val][2][sal] for val,sal in zip(data['lat'],data['dayofyear'])]
data['lengthofdayd10'] = [qryToDL[val][3][sal] for val,sal in zip(data['lat'],data['dayofyear'])]

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

largerolling_n = 7*8
shortrolling_n = 7*2

container = []
dataframes = []
coefs = []
selectedlats = []
dataqrys = []

for k,val in enumerate(finalqrys):
    
    selected = newdata[newdata['qry']==val]
    largerolling = selected.groupby('date').mean(numeric_only=True).rolling(largerolling_n).mean()
    disc = largerolling.corr()['cases'].abs()
    nans = largerolling.isna().sum()
        
    for inx in nans.index:
         if nans[inx] >int(1.15* largerolling_n):
             disc[inx] = 0
        
    cLat = selected['lat'].mean()
    cQry = selected['qry'].iloc[0]
    
    shortrolling = selected.groupby('dayofyear')['cases'].apply(np.median)
    shortrolling = shortrolling.to_frame()
    
    lengths = [GetDayLenght(J, cLat) for J in range(0,366)]
    disc_range = [k for k in range(172,192)]
    
    for i in range(5,20):
        
        corder = i/20
        diff_dl = df.GL(corder,lengths,num_points=len(lengths))
        shortrolling['fractional'] = diff_dl[shortrolling.index-1]
        shortrolling['norm_fractional'] = (shortrolling['fractional']-shortrolling['fractional'].min())/(shortrolling['fractional'].max()-shortrolling['fractional'].min())
        short_disc = shortrolling.groupby('norm_fractional')['cases'].mean().rolling(shortrolling_n).mean().argmax()
        
        if short_disc in disc_range:
            coefs.append(corder)
            selectedlats.append(cLat)
            dataframes.append(shortrolling)
            container.append(disc)
            dataqrys.append(cQry)
            break
        
###############################################################################
# Loading packages 
###############################################################################

fig = plt.figure(figsize=(25,15))
gs = gridspec.GridSpec(nrows=8, ncols=10) 

axs = [fig.add_subplot(gs[0,k]) for k in range(10)]

locationdata = newdata[newdata['qry'].isin(dataqrys)]
gr_locationdata = locationdata.groupby('qry').mean(numeric_only=True)

exampledata = locationdata[locationdata['qry']=='lat==19.482945 & long==-99.113471']
exampledata = exampledata.groupby('dayofyear').mean(numeric_only=True)

ex_lengths = [GetDayLenght(J, exampledata['lat'].mean()) for J in range(0,366)]
ex_colors = [plt.cm.Blues(val) for val in np.linspace(0.5,1,num=10)]

for k in range(10):
    ex_dl = df.GL((k+1)/10,ex_lengths,num_points=len(lengths))
    exampledata['fractional'] = ex_dl[exampledata.index-1]
    exampledata['norm_fractional'] = (exampledata['fractional']-exampledata['fractional'].min())/(exampledata['fractional'].max()-exampledata['fractional'].min())
    exrolling = exampledata.groupby('norm_fractional')['cases'].mean().rolling(shortrolling_n).mean()
    exrolling.plot(ax=axs[k],color=ex_colors[k],label=str((k+1)/10)+'SD')
    axs[k].set_ylim([0,1100])
    axs[k].set_xlabel('Normalized \n Fractional Order',fontsize=14)
    axs[k].legend(loc=1)
    PlotStyle(axs[k])

axs[0].text(0.01, 0.99, 'A', size=25, color='black', ha='left', va='top', transform=axs[0].transAxes)

axmp = fig.add_subplot(gs[1:6,0:7])

m = Basemap(projection='cyl',llcrnrlat=-65, urcrnrlat=80,
            llcrnrlon=-180, urcrnrlon=180,ax=axmp)
m.drawcoastlines(color='gray')
m.fillcontinents(color='gainsboro')
m.drawcountries(color='gray')
parallels = np.arange(-80,80,10.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,True,True,False],alpha=0.5)
sctr = m.scatter(gr_locationdata['long'].values,gr_locationdata['lat'].values,c=coefs,
          s=gr_locationdata['cases'].values/5,cmap='Blues_r')
axmp.set_frame_on(False)
cbar = plt.colorbar(sctr,location="bottom",aspect=50)
cbar.set_label('Fractional Order',fontsize=fontsize)
axmp.text(0.01, 0.99, 'B', size=25, color='black', ha='left', va='top', transform=axmp.transAxes)

axlats = fig.add_subplot(gs[2:8,7:10])

lat_colors = [plt.cm.Blues(val) for val in np.linspace(0.5,1,num=len(selectedlats))]

for i,(lt, dfr) in enumerate(zip(selectedlats,dataframes)):
    
    localdf = dfr.groupby('norm_fractional')['cases'].mean().rolling(shortrolling_n).mean()
    localdf = (localdf - localdf.min())/(localdf.max() - localdf.min())
    
    axlats.plot(np.array(localdf.index),(lt+15*np.array(localdf)),alpha=0.05,color=lat_colors[i])

axlats.set_ylabel('Normalized Cases By Latitude',fontsize=14)
axlats.set_xlabel('Normalized \n Fractional Order',fontsize=14)
axlats.text(0.01, 0.99, 'D', size=25, color='black', ha='left', va='top', transform=axlats.transAxes)
PlotStyle(axlats)
        
fulldata = pd.concat(dataframes)
groupdata = fulldata.groupby(pd.cut(fulldata['norm_fractional'],100)).mean()
inx = [val.right for val in groupdata.index]

axmean = fig.add_subplot(gs[1,7:10])
axmean.plot(inx,np.array(groupdata['cases']),color='navy',label='Mean Cases')

axmean.set_ylabel('Normalized \n Cases',fontsize=14)
axmean.set_xlabel('Normalized \n Fractional Order',fontsize=14)
axmean.legend()
axmean.text(0.01, 0.99, 'C', size=25, color='black', ha='left', va='top', transform=axmean.transAxes)
PlotStyle(axmean)

ndf = pd.concat(container,axis=1)
ndf.columns = ['qry_' + str(k) for k in range(len(container))]

ndfmean = ndf.T.mean()

datacolumns = ['dayofyear','lengthofday','lengthofdayd03','lengthofdayd06',
               'lengthofdayd10','spots','UVA', 'UVB', 'UVC', 'Vis', 'NIR',
               'SWIR','normspots','normUVC','normVis','normSWIR']

datanames = ['DOY','SD','0.3SD','0.6SD','SDRC','NS','UVA', 'UVB', 'UVC', 
             'Vis', 'NIR','SWIR','NrNS','NrUVC','NrVis','NrSWIR']

cmap = plt.cm.Blues
histaxs = [fig.add_subplot(gs[6+kk,ii]) for ii in range(7) for kk in range(2)]

for k,cl in enumerate(histaxs):
    
    cnts, values, bars = histaxs[k].hist(ndf.T[datacolumns[k]].values,
                                         bins=50,label=datanames[k])
    histaxs[k].set_ylim([0,350])
    histaxs[k].set_xlim([0,1])
    
    alp = values[np.argmax(cnts)]
    
    if (1.25)*alp>=1:
        alp = 1
    else:
        alp=1.25*alp
    
    for i, (cnt, value, bar) in enumerate(zip(cnts, values, bars)):
        bar.set_facecolor(cmap(cnt/cnts.max(),alpha=alp))
    
    histaxs[k].set_xlabel('Pearson \n correlation',fontsize=14)
    histaxs[k].set_ylabel('Frequency',fontsize=14)
    histaxs[k].legend()
    PlotStyle(histaxs[k])

histaxs[0].text(0.01, 0.99, 'E', size=25, color='black', ha='left', va='top', transform=histaxs[0].transAxes)
plt.tight_layout()

plt.savefig('/media/tavo/storage/images/image_map.png',dpi=75,bbox_inches='tight')
