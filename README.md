# hyperpyper
[Bernhard Lehner](https://www.researchgate.net/profile/Bernhard_Lehner)

Insights with Interventions via Transforms:
- Automatic collation of full batch output from transformation pipelines.
- Tools to load, cache, and plot transformation pipeline outputs.
- Specific transforms for interventions such as single pixel modifications, simulating compression artifacts, dim. reduction.
- Wrapper for logging/debugging transformation pipelines.
- Demos and Tutorials

## Prerequisites
Tested with Python 3.8, 3.9, and 3.10 on Windows/Linux.

## Installation
pip install git+https://github.com/berni-lehner/hyperpyper.git


## Abstract
todo

![alt text](https://github.com/berni-lehner/hyperpyper/blob/main/meta/embedding_plotter.png?raw=true)

## Table of Contents
1. [Introduction](#introduction)
1. [Demos](#demos)
1. [Citation](#citation)


## Introduction <a name="introduction"></a>
todo

## Demos <a name="demos"></a>
The notebooks contain demos and tutorials for several usecases where ```hyperpyper``` is utilized.

- **Image processing**
    - [Extracting histograms from CIFAR10](https://github.com/berni-lehner/hyperpyper/blob/main/CIFAR10_hist_demo.ipynb)
    - [Extracting features from CIFAR10](https://github.com/berni-lehner/hyperpyper/blob/main/CIFAR10_features_demo.ipynb)

- **Utility classes**
    - [EmbeddingPlotter](https://github.com/berni-lehner/hyperpyper/blob/main/EmbeddingPlotter_demo.ipynb)
    - [HistogramPlotter](https://github.com/berni-lehner/hyperpyper/blob/main/HistogramPlotter_demo.ipynb)
    - [MultiFigurePlotter](https://github.com/berni-lehner/hyperpyper/blob/main/MultiFigurePlotter_demo.ipynb)    
    - [DataSetDumper](https://github.com/berni-lehner/hyperpyper/blob/main/DataSetDumper_demo.ipynb)
    - [FolderScanner](https://github.com/berni-lehner/hyperpyper/blob/main/FolderScanner_demo.ipynb)
    - [PathList](https://github.com/berni-lehner/hyperpyper/blob/main/PathList_demo.ipynb)
    - [Pickler](https://github.com/berni-lehner/hyperpyper/blob/main/Pickler_demo.ipynb)

- **Pytorch demos**
    - [Caching embeddings and UMAP](https://github.com/berni-lehner/hyperpyper/blob/main/CIFAR10_umap_caching_demo.ipynb)

- **sklearn demos**


## Citation <a name="citation"></a>
If you find this code useful in your research, please cite:
    
...



## Acknowledgements
Thanks to chuber1986 for implementing the proof-of-concept of the image preview in the EmbeddingPlotter!
