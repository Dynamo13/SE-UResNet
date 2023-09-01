# SE-UResNet
This repository contains the official codebase of SE-UResNet which was accepted in AIHC workshop 2023 of IEEE IRI conference [Paper](https://www.computer.org/csdl/proceedings-article/iri/2023/345800a261/1Q259zGjbH2). SE-UResNet focuses on segmenting multiple organs/regions of interests such as Heart, Lungs, Trachea, Collarbone, and Spine from chest radiograph images.

## Overview
A novel fully connected segmentation model which provides a solution to problem of multiorgan segmentation from Chest X-Rays by incorporating a novel residual module. This module in conjuction with S&E modules in individual decoding pathway improves segmentation accuracy and also makes this model reproducible. The implementation is inspired from ***Attention UW-Net: A fully connected model for automatic segmentation and annotation of chest X-ray***  [Code](https://github.com/Dynamo13/Attention_UWNet) | [Paper]( https://www.sciencedirect.com/science/article/abs/pii/S0010482522007910).

## Network Architecture
## Datasets
The datasets used in the paper can be downloaded from the links below:
- [VinDr-RibCXR](https://vindr.ai/datasets/ribcxr)
- [JSRT](http://db.jsrt.or.jp/eng.php)
- [Shenzen](https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask)
- [NIH CXR](https://www.kaggle.com/datasets/nih-chest-xrays/data)

  In order to download the segmentation masks please refer to the [link]()

  ## Citation
  ```
  D. Pal, T. Meena and S. Roy, "A Fully Connected Reproducible SE-UResNet for Multiorgan Chest Radiographs Segmentation," 2023 IEEE 24th International Conference on Information Reuse and Integration for Data Science (IRI), Bellevue, WA, USA, 2023, pp. 261-266, doi:     
  10.1109/IRI58017.2023.00052.
  ```
