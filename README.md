# SARS-Cov2 seasonality and adaptation are driven by solar activity.

## Manuscript
A copy of the manuscript can also be found at the paper folder
- https://www.researchsquare.com/article/rs-2797280/v2

## Summary 
A summary of the findings can be found here 
- https://www.kaggle.com/code/tavoglc/a-computational-description-of-sarscov2-adaptation

## Request
If you for any reason clone this repo please consider leaving any comments at pubpeer https://pubpeer.com/, under the following doi doi: 10.21203/rs.3.rs-2797280/v3 (https://pubpeer.com/publications/B48745C2D18A5E3589992677D8E055)That will allow me to have all the comments in a single place and to further improve the results described in both the code and the manuscript. 

## Third draft
Third version was the result of data update and a series of changes listed below.  

- Results from all models are evaluated using the same samples, removing possible dta leaks from previous versions. But lowers the amoun of data for other analysis. 
- Most of the naming problems are solved on the model-generating functions. 
- Data was updated to contain sequences up to July 2023. 
- A new preprocessing step was added sequences were cropped from the first start codon to the last stop codon, the resulting genome is referred to as the effective genome. 
- The reversed graph is removed from the previous version of the relational model.
- Two new datasets are added, relational graphs and effective genomes.


## Second Draft

### Models

Training the dimensional expansion model
- https://www.kaggle.com/code/tavoglc/dimensional-expansion-for-sars-cov2-genome-data

### Trained Models
Contains the trained models IDs for each fold and scalers. 
- https://www.kaggle.com/datasets/tavoglc/sarscov2-jax-models

Trained models are broken for the generative part, yet they can be used for dimensionality reduction. A new version can be found down here, the generative part of the model is fixed as well as some naming errors.  There's no need to modify the frozen dictionaries to use the model. 
Loss is changed to obtain a calibrated probability to ease the inference. 

- https://www.kaggle.com/code/tavoglc/generative-covid19-genomes

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

## Extras 

### Surveillance probes. 
#### Kmer model
- https://www.kaggle.com/code/tavoglc/probes-for-sarscov2-weekly-surveillance

#### Relational Model

#### Full genome model


### Full genome prediction. 
- https://www.kaggle.com/code/tavoglc/generative-covid19-genomes

### Full effective genome prediction. 
- https://www.kaggle.com/code/tavoglc/generative-effective-covid19-genomes

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


