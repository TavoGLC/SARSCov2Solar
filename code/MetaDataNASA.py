#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:51:40 2023

@author: tavo
"""

###############################################################################
# Loading packages 
###############################################################################

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.interpolate import RectSphereBivariateSpline


###############################################################################
# Loading packages 
###############################################################################

dataorg = '/media/tavo/storage/nasa_npy/'
datadir = dataorg + 'data/'

files = np.sort(os.listdir(datadir))
filePaths = [datadir+val for val in files]

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covid/datasets/MetaData.csv')
MetaData = MetaData.set_index('id')

uniqueDates = MetaData['date'].unique()

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
    
    cop = MetaData[MetaData['date']==dte].copy()
    
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

newMD = pd.concat(toConcat)
newMD.to_csv('/media/tavo/storage/biologicalSequences/covid/datasets/MetaDataNASA.csv')
