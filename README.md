# QCNN-Fold
This repository is about quantum convolutional neural network in protein distance prediction. The code is based on pdnet project: https://github.com/ba-lab/pdnet.

>This repository contains the official implementation of the following paper:
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
