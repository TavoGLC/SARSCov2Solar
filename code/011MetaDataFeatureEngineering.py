#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 18:51:19 2023

@author: tavo
"""

import os
import numpy as np
import pandas as pd
import differint.differint as df

from scipy.interpolate import RectSphereBivariateSpline

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


def ReadFile(path):    
    with open(path) as f:
        lines = f.readlines()
    return str(lines[0])

###############################################################################
# Loading packages 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData.csv')
finalentrys = os.listdir('/media/tavo/storage/biologicalSequences/covidsr04/sequences/single/')
finalids = [val[0:-4] for val in finalentrys]

MetaData = MetaData[MetaData['Accession'].isin(finalids)]
MetaData['id'] = MetaData['Accession']

MetaData['effectivelength'] = [len(ReadFile('/media/tavo/storage/biologicalSequences/covidsr04/sequences/single/'+val+'.txt')) for val in MetaData['id']]

###############################################################################
# Loading packages 
###############################################################################

locationsdf = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/locations.csv')
locationsdf = locationsdf.set_index('name')
locsdata = locationsdf.loc[MetaData['Geo_Location']].values

MetaData['lat'] = locsdata[:,0]
MetaData['lon'] = locsdata[:,1]

MetaData['query'] = ['lat=='+str(val)+' & long=='+str(sal)  for val,sal in zip(MetaData['lat'],MetaData['lon'])]

###############################################################################
# Loading packages 
###############################################################################

MetaData['date'] = pd.to_datetime(MetaData['Collection_Date'])
MetaData = MetaData[MetaData['date'] < '2023-6-05']

MetaData['dayofyear'] = MetaData['date'].dt.dayofyear
MetaData['dayofweek'] = MetaData['date'].dt.dayofweek
MetaData['week'] = MetaData['date'].dt.isocalendar().week

MetaData['outbreakdays'] = (MetaData['date'] - MetaData['date'].min()).dt.days

###############################################################################
# Loading packages 
###############################################################################

uniquequery = MetaData['query'].unique()
lats = MetaData.groupby('query')['lat'].mean()

qrytoduration = {}
qrytodelta = {}

for qry,lat in zip(lats.index,lats.values):
    
    day = {}
    delta = {}
    
    daydurations = GetYearLengths(lat)
    durationdelta = df.GL(1,daydurations,num_points=len(daydurations))
    
    for kk in range(len(daydurations)):
        day[kk] = daydurations[kk]
        delta[kk] = durationdelta[kk]
    
    qrytoduration[qry] = day
    qrytodelta[qry] = delta
    
MetaData['daylength'] = [qrytoduration[qr][dy] for qr,dy in zip(MetaData['query'],MetaData['dayofyear'])]
MetaData['daylengthd10'] = [qrytodelta[qr][dy] for qr,dy in zip(MetaData['query'],MetaData['dayofyear'])]

###############################################################################
# Loading packages 
###############################################################################

datasolar = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/solar/sunspots2023.csv')
datasolar['date'] = pd.to_datetime(datasolar['date'])
datasolar = datasolar.set_index('date')

MetaData['sunspots'] = datasolar['dailysunspots'].loc[MetaData['date']].values

###############################################################################
# Loading packages 
###############################################################################

dataradiation = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/solar/solarcurrent2023.csv')
dataradiation['date'] = pd.to_datetime(dataradiation['date'])
dataradiation = dataradiation.set_index('date')

for col in dataradiation.columns:
    MetaData[col] = dataradiation[col].loc[MetaData['date']].values

###############################################################################
# Loading packages 
###############################################################################

datadir = '/media/tavo/storage/biologicalSequences/covidsr04/data/NASA/AIRSNPY/data/'

files = np.sort(os.listdir(datadir))
filePaths = [datadir+val for val in files]

uniqueDates = MetaData['date'].unique()

###############################################################################
# Loading packages 
###############################################################################

lats = np.linspace(0.1, 179, 180) * np.pi / 180
lons = np.linspace(0.1, 359, 360) * np.pi / 180

toConcat = []

names = ['SurfPres_Forecast_A','SurfSkinTemp_A','SurfAirTemp_A','CloudFrc_A',
         'TotO3_A','OLR_A','ClrOLR_A','TotH2OVap_A','RelHumSurf_A',
         'SurfPres_Forecast_D','SurfSkinTemp_D','SurfAirTemp_D','CloudFrc_D',
         'TotO3_D','OLR_D','ClrOLR_D','TotH2OVap_D','RelHumSurf_D']

for dte in uniqueDates:
    
    cop = MetaData[MetaData['date']==dte].copy()
    
    lat = ((cop['lat']+90)*np.pi/180).values
    long = ((cop['lon']+180)*np.pi/180).values
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

dcolumns = ['Submitters', 'Organization','Org_location', 'Pangolin', 'outbreakdays',
            'PangoVersions','Isolate', 'Species', 'Length', 'Geo_Location', 
            'Host', 'Isolation_Source','BioSample', 'id', 'effectivelength', 
            'lat', 'lon','query', 'date', 'dayofyear', 'dayofweek', 'week', 
            'daylength','daylengthd10', 'sunspots', 'UVA', 'UVB', 'UVC', 
            'Vis', 'NIR', 'SWIR','SurfPres_Forecast_A', 'SurfSkinTemp_A', 
            'SurfAirTemp_A', 'CloudFrc_A','TotO3_A', 'OLR_A', 'ClrOLR_A', 
            'TotH2OVap_A', 'RelHumSurf_A','SurfPres_Forecast_D', 
            'SurfSkinTemp_D', 'SurfAirTemp_D', 'CloudFrc_D','TotO3_D', 
            'OLR_D', 'ClrOLR_D', 'TotH2OVap_D', 'RelHumSurf_D']

toConcat = toConcat[dcolumns]
toConcat.to_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData2023.csv',index=False)
