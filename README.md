# SARS-Cov2 seasonality and adaptation are driven by solar activity.

## Manuscript
A copy of the manuscript can also be found at the paper folder
- https://www.researchsquare.com/article/rs-2797280/v2

## Second Draft

### Models

Training the dimensional expansion model
- https://www.kaggle.com/code/tavoglc/dimensional-expansion-for-sars-cov2-genome-data

### Trained Models
Contains the trained models IDs for each fold and scalers. 
- https://www.kaggle.com/datasets/tavoglc/sarscov2-jax-models

### Figures
There's a bug that rises due to how the layers are named. Figure 5 provides an example of how to modify the frozen dict to be able to use the model. Also, each name declaration will raise an error, I'm using the same flax and jax versions as Kaggle, so I don't think is due to that. That error can be solved by changing name to Name as name is restricted in flax.  

Figure 5
- https://www.kaggle.com/code/tavoglc/sars-cov-2-genome-wise-variability/notebook

Figure 7
- https://www.kaggle.com/code/tavoglc/solar-patterns-and-sarscov2-genome-size

Figure 9 
- https://www.kaggle.com/code/tavoglc/environment-and-covid-19

Figure 11
- https://www.kaggle.com/code/tavoglc/dynamical-vaes-for-sars-cov2

Figure 13
- https://www.kaggle.com/code/tavoglc/sars-cov2-genome-adaptation-and-the-environment

### Datasets
Reconstructed NASA Aqua/AIRS L3 Daily Standard Physical Retrieval 
- https://www.kaggle.com/datasets/tavoglc/nasa-aquaairs-l3-reconstructed

Nasa files visualization 
- https://www.tiktok.com/@tavoglc0?lang=es

## First Draft 
### Notebooks

#### Different examples of MLPs applied to frequency data implemented in Tensorflow and Keras
- https://www.kaggle.com/code/tavoglc/sars-cov-2-variational-autoencoders-with-k-mers
- https://www.kaggle.com/code/tavoglc/seasonal-disentangling-of-sars-cov-2

#### MLP applied to frequency data implemented in JAX and Flax
- https://www.kaggle.com/code/tavoglc/autoencoders-jax-and-sars-cov2

#### Model inference example
- https://www.kaggle.com/code/tavoglc/sars-cov2-mutational-hotspots

### Datasets
#### Trained Models 
Sufix indicates the fold, CSV file contains the sequence ids for each fold
- https://www.kaggle.com/datasets/tavoglc/sarscov2-trained-models-fragments

#### Meta Data
Geo location and other features 
- https://www.kaggle.com/datasets/tavoglc/covid19-metadata

#### Fragment frequencies small format. 
- https://www.kaggle.com/datasets/tavoglc/sars-cov2-fragments-frequency-small

#### Fragment frequencies large format. 
- https://www.kaggle.com/datasets/tavoglc/covid19-sequences-extended?select=KmerDataUpd.csv

#### Cases Data

##### Europe
- https://www.kaggle.com/datasets/tavoglc/covid19-cases
##### America
- https://www.kaggle.com/datasets/tavoglc/covid19-in-the-american-continent

### Figures
#### Figure 1 and 4
- https://www.kaggle.com/code/tavoglc/sars-cov2-seasonality-and-adaptation

#### Figure 2
- https://www.kaggle.com/code/tavoglc/sars-cov2-seasonality-and-adaptation-02

#### Figure 3
- https://www.kaggle.com/code/tavoglc/sars-cov2-seasonality-and-adaptation-03

## Want to know more ? 
### Sliding sampling 
#### Paper
- https://www.researchsquare.com/article/rs-1691291/v1
#### Repo 
- https://github.com/TavoGLC/SlidingSampling

### Old medium posts 
- https://medium.com/@tavoglc/list/machine-learning-and-covid19-0659a2c0bb92
### Substack 
- https://tavoglc.substack.com/


