# QCNN-Fold

> **Quantum Convolutional Neural Network on Protein Distance Prediction**
> 
> Accepted by the International Joint Conference on Neural Networks (IJCNN) 2021, Oral

## Introduction

This repository is the official implementation of [Quantum Convolutional Neural Network on Protein Distance Prediction](https://ieeexplore.ieee.org/document/9533405).

<div align=center><img src="https://github.com/zhenhouhong/QCNN-Fold/blob/main/qcnn-fold-process.png"></div>

<div align='center'>The major part of QCNN in protein inter-residue distance prediction problem.</div>

## Requirements
- Linux (Test on Ubuntu18.04)
- Python3.6+ (Test on Python3.6.8)
- PyTorch
- PennyLane
- Librosa (version 0.7.2)
- Numba (version 0.48.0)

## Basic framework
- qcircuit: the variational quantum circuit(VQC) and hybrid VQC.
- qconv: the quantum convolutional layer.
- qmodels: the qcnn models, contains the Basic-QCNN, QCNN-RDD, QCNN-RDD-distance.

 ### Notes
 Except the qcircuit.py, qconv.py, and qmodels.py, another part of code is based on the [pdnet](https://github.com/ba-lab/pdnet) project.

## How to use

### Download the datasets
- http://deep.cs.umsl.edu/pdnet/

### Train the QCNN models
- Setting the config by `python3 train.py -h`
- Edit the train.py
- `python3 train.py`

## Citation
If you find QCNN-Fold useful in your research, please consider citing:

    @inproceedings{hong2021quantum,
      title={Quantum Convolutional Neural Network on Protein Distance Prediction},
      author={Hong, Zhenhou and Wang, Jianzong and Qu, Xiaoyang and Zhu, Xinghua and Liu, Jie and Xiao, Jing},
      booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
      pages={1--8},
      year={2021},
      organization={IEEE}
    }
 
