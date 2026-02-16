# Expert Models Training Guide

Created by Noas Shaalan,

This directory contains the scripts used to prepare and train the expert models for ARES, LMMS, and SensEmBERT.

Note: We have provided the extracted features as .pt files. However, in order to recreate these features, you must first download the pre-trained vectors from the sources listed below.

### ARES Expert
Vectors Source: https://nlp.uniroma1.it/sensembert/

To run:
1. Extract features: `python ares_expert/prep_features_final.py`
2. Train model: `python ares_expert/train_mlp_final.py`

### LMMS Expert
Vectors Source: https://figshare.com/articles/dataset/LMMS_2019_/21977219

To run:
1. Extract features: `python lmms_expert/prep_features_final.py`
2. Train model: `python lmms_expert/train_mlp_final.py`

### SensEmBERT Expert
Vectors Source: https://nlp.uniroma1.it/sensembert/

To run:
1. Extract features: `python sensembert_expert/prep_features_final.py`
2. Train model: `python sensembert_expert/train_mlp_final.py`

### General Instructions
- Ensure your environment is set up with PyTorch and the required dependencies.
- All scripts use shared utilities from the parent directory.
- Before running the training scripts, ensure the context features have been extracted or are already present in the folder.
