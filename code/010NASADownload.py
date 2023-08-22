#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 00:00:36 2023

@author: tavo
"""


import time 
import requests

# Set the URL string to point to a specific data URL. Some generic examples are:
#   https://servername/data/path/file
#   https://servername/opendap/path/file[.format[?subset]]
#   https://servername/daac-bin/OTF/HTTP_services.cgi?KEYWORD=value[&KEYWORD=value]

urls_file = '/media/tavo/storage/subset_OMUVBd_003_20230811_055632_.txt'

with open(urls_file) as f:
    urls = f.readlines()

for ur in urls[300::]:
    time.sleep(4)
    URL = ur
    # Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name. 
    FILENAME ='/media/tavo/storage/biologicalSequences/covidsr04/data/NASA/OMUV/' + URL[122:138] + '.hdf.nc4'
    #FILENAME ='/media/tavo/storage/biologicalSequences/covidsr04/data/NASA/OMUV/' + URL[82:100] + '.hdf.nc4'
          
    result = requests.get(URL)
    try:
        result.raise_for_status()
        f = open(FILENAME,'wb')
        f.write(result.content)
        f.close()
        print('contents of URL written to '+FILENAME)
    except:
        print('requests.get() returned an error code '+str(result.status_code))
        
