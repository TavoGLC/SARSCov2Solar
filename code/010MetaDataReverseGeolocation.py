#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:58:15 2023

@author: tavo
"""
import time
import string
import pandas as pd

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

###############################################################################
# Loading packages 
###############################################################################

MetaData = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/MetaData.csv')
locations = MetaData['Geo_Location'].unique()
locations = [str(val) for val in locations]
cleanlocations = []

coords = []
exclude = set(string.punctuation)

for loc in locations:
    tst = []
    for k in range(len(loc)):
        if loc[k] in [':','/',',']:
            if k+1<len(loc):
                if loc[k+1]==' ':
                    pass
                else:
                    tst.append(' ')
        else:
            tst.append(loc[k])
                
    s = ''.join(tst)
    cleanlocations.append(s)

###############################################################################
# Loading packages 
###############################################################################

geoCoords = []
geolocator = Nominatim(user_agent="locations")
geocode = RateLimiter(geolocator.geocode,min_delay_seconds=5)
coords = []

for val in cleanlocations:
    
    location = geocode(val) 
    if type(location)==type(None):
        
        coords.append((0,0,0))
        time.sleep(2.5)
        
    else:
        
        lat = location.latitude
        long = location.longitude
        alt = location.altitude
        
        coords.append((lat,long,alt))
        time.sleep(2.5)
        
###############################################################################
# Loading packages 
###############################################################################

locationsdf = pd.DataFrame()
locationsdf['name'] = locations
locationsdf['lat'] = [val[0] for val in coords]
locationsdf['lon'] = [val[1] for val in coords]
locationsdf['alt'] = [val[2] for val in coords]

locationsdf.to_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/locations.csv',index=False)
