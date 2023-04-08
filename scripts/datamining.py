#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:40:48 2023

@author: tavo
"""

import mygene
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

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
# Visualization functions
###############################################################################

file = '/media/tavo/storage/biologicalSequences/covid/datasets/referenceTranscrips/fold_out_0.txt'
transcriptsdir = '/media/tavo/storage/biologicalSequences/covid/datasets/referenceTranscrips/'

with open(file) as f:
    lines = f.readlines()

lines = [ln.strip() for ln in lines]

mg = mygene.MyGeneInfo()

results = mg.querymany(lines, scopes='refseq.rna',fields='all',as_dataframe=True)

###############################################################################
# Visualization functions
###############################################################################

dise = results['clingen.clinical_validity'].dropna()
cont = []
for val in dise:
    for sal in val:
        cont.append(sal['disease_label'])
disease = pd.DataFrame(cont).value_counts(sort=False)

fig = plt.figure(figsize=(25,30))
gs = gridspec.GridSpec(nrows=2, ncols=2)

axs0 = fig.add_subplot(gs[0,0])

disease.plot.bar(ax=axs0)
axs0.set_ylabel('Frequency',fontsize=fontsize)
axs0.text(0.01, 0.99, 'A', size=25, color='black', ha='left', va='top', transform=axs0.transAxes)
PlotStyle(axs0)

###############################################################################
# Visualization functions
###############################################################################

tt = []
intp = results['interpro'].dropna()
cont = []
for k,val in enumerate(intp):
    for sal in val:
        cont.append(sal['desc'])
        if sal['desc']=='Immunoglobulin-like fold':
            tt.append(k)

intpro = pd.DataFrame(cont).value_counts(sort=False)

axs1 = fig.add_subplot(gs[0,1])
intpro[intpro>70].plot.bar(ax=axs1)
axs1.set_ylabel('Frequency',fontsize=fontsize)
axs1.text(0.01, 0.99, 'B', size=25, color='black', ha='left', va='top', transform=axs1.transAxes)
PlotStyle(axs1)

###############################################################################
# Visualization functions
###############################################################################

rer = results['pathway.reactome'].dropna()
contr = []
for val in rer:
    for sal in val:
        contr.append(sal['name'])

react = pd.DataFrame(contr).value_counts(sort=False)

axs2 = fig.add_subplot(gs[1,:])
react[react>100].plot.bar(ax=axs2)
axs2.set_ylabel('Frequency',fontsize=fontsize)
axs2.text(0.01, 0.99, 'C', size=25, color='black', ha='left', va='top', transform=axs2.transAxes)
PlotStyle(axs2)
plt.tight_layout()
plt.savefig('/media/tavo/storage/images/image_dm.png',dpi=75,bbox_inches='tight')

###############################################################################
# Visualization functions
###############################################################################

dataList = results['pharmgkb'].dropna().values

outfile = transcriptsdir+'ids'+str(0)+'.txt'

with open(outfile, 'w') as f:
    for line in dataList:
        f.write(line)
        f.write('\n')
        
dataList = results['uniprot.Swiss-Prot'].dropna().values
uids = []
for val in dataList:
    if type(val)==str:    
        uids.append(val)
    else:
        for sal in val:
            uids.append(sal)
uids = list(set(uids))
outfile = transcriptsdir+'ids_up'+str(1)+'.txt'

with open(outfile, 'w') as f:   
    for line in uids:
        if type(line)==str:    
            f.write(line)   
            f.write('\n')
        else:
            f.write(line[0])   
            f.write('\n')
        
