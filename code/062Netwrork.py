#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:58:53 2023

@author: tavo
"""

import numpy as np
import pandas as pd 
import networkx as nx

import matplotlib.pyplot as plt

###############################################################################
# Loading packages 
###############################################################################

file = '/media/tavo/storage/biologicalSequences/covidsr04/data/selected/fold_out_gene.txt'

with open(file) as f:
    lines = f.readlines()

lines = [ln.strip() for ln in lines]

data = pd.read_csv('/media/tavo/storage/biologicalSequences/covidsr04/data/selected/string_interactions.tsv',sep='\t')

unique_nods = list(set(data['#node1'])) + list(set(data['node2']))
unique_nods = list(set(unique_nods))

nameToNumber = {}
numberToName = {}

for k,val in enumerate(unique_nods):
    nameToNumber[val] = k
    numberToName[k] = val


data['nodeA'] = [nameToNumber[val] for val in data['#node1']]
data['nodeB'] = [nameToNumber[val] for val in data['node2']]

notInQuery = [val for val in unique_nods if val not in lines]
notInQuery = list(set(notInQuery))

inQuery = [val for val in unique_nods if val in lines]
inQuery = list(set(inQuery))

###############################################################################
# Loading packages 
###############################################################################

G = nx.Graph()
G.add_nodes_from([k for k in range(len(unique_nods))])
G.add_edges_from([(val,sal) for val,sal in zip(data['nodeA'],data['nodeB'])])

###############################################################################
# Loading packages 
###############################################################################

neighbours = {n: len(list(nx.all_neighbors(G, n))) for n in G.nodes}
threshold = 5 

selectednodes = [k for k in range(559) if neighbours[k]>threshold]

notIn = [nameToNumber[val] for val in notInQuery]
notIn = [val for val in notIn if val in selectednodes]

inSelection = [nameToNumber[val] for val in inQuery]
inSelection = [val for val in inSelection if val in selectednodes]

G0 = nx.Graph()
G0.add_nodes_from(selectednodes)
G0.add_edges_from([(val,sal) for val,sal in zip(data['nodeA'],data['nodeB']) if val in selectednodes and sal in selectednodes])

cent = nx.eigenvector_centrality(G0)

highcent = [val for val in cent.keys() if cent[val] > 0.135]

labs = {}

for val in highcent:
    labs[val]=numberToName[val]

plt.figure(figsize=(10,10))
ax = plt.gca()

pos = nx.circular_layout(G0)
nx.draw_networkx_edges(G0,pos=pos,alpha=0.075)
nx.draw_networkx_nodes(G0,pos=pos,nodelist=notIn,
                       node_color='red',node_size=[500*cent[val] for val in notIn],alpha=0.75)
nx.draw_networkx_nodes(G0,pos=pos,nodelist=inSelection,
                       node_color='blue',node_size=[500*cent[val] for val in inSelection],alpha=0.75)
nx.draw_networkx_nodes(G0,pos=pos,nodelist=highcent,
                       node_color='grey',node_size=[1000*cent[val] for val in highcent],alpha=0.5)

for val in highcent:
    x,y = pos[val]
    rot = np.degrees(np.arctan(y/x))     
    if x<0:
        x = x-0.12
    if y<0:
        y = y-0.12
    
    ax.text(x,y,numberToName[val],rotation=rot)

ax.axis('off')
plt.savefig('/media/tavo/storage/biologicalSequences/covidsr04/data/images/image_intr.png',dpi=75,bbox_inches='tight')

