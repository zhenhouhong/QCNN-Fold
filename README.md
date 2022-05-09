# QCNN-Fold
This repository is about quantum convolutional neural network in protein distance prediction. Except the qcircuit.py, qconv.py and qmodels.py, other part of code is based on [pdnet](https://github.com/ba-lab/pdnet) project.

>This repository contains the official implementation of the following [paper](https://ieeexplore.ieee.org/document/9533405):
>
>**Quantum Convolutional Neural Network on Protein Distance Prediction**

# Install
- pip install torch
- pip install pennylane
- pip install librosa==0.7.2
- pip install numba==0.48.0

# Basic framework
- qcircuit: the variational quantum circuit(VQC) and hybrid VQC.
- qconv: the quantum convolutional layer.
- qmodels: the qcnn models, contains the Basic-QCNN, QCNN-RDD, QCNN-RDD-distance.

# How to use

## Download the datasets
- http://deep.cs.umsl.edu/pdnet/

## Train the QCNN models
- Setting the config by `python3 train.py -h`
- Edit the train.py
- `python3 train.py`
