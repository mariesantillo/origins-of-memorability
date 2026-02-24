# The Origins of Memorability in Infancy

This repository contains the full analysis pipeline for the manuscript:

**“The Origins of Memorability in Infancy”**, investigating how memorability estimated by deep neural networks such as ResMem (Needell & Bainbridge, 2022) predict neural responses in two month old infants and adults during awake naturalistic video fMRI.

---

##  Overview

This project tests whether frame-level memorability scores derived from the ResMem model predict brain activity during fMRI scanning.

The pipeline includes: 

1. Frame extraction from videos (.mp4)
2. Feature computations such as visual entropy, attention saliency and memorability on the video frames
3. Generation of parametric modulation table
4. First-level GLM analyses
5. Second-level permutation testing (TFCE) using fslrandomise
6. ROI analysis

---

## Feature Computation 
All the code to calculate the different features can be found in memorability/feature_computation/ with one subfolder for each feature.
### ResMem Memorability Model
Memorability scores were computed using the pretrained ResMem model (https://github.com/Brain-Bridge-Lab/resmem).
Each video was extracted into frames (25 fps) and scored frame-by-frame, predictions were then averaged over 15 frames to match with 0.61 TR resolution
### Entropy Calculation 
Visual entropy scores were computed with VCA (https://github.com/cd-athena/VCA)) using 32x32 block sizes and scores were averaged every 15 frames. 
### Saliency calculation 
Saliency predictions were generated using STRA-Net (https://github.com/ashleylqx/STRA-Net). Optic flow images were computed with RAFT (https://github.com/princeton-vl/RAFT) and fed into the STRANET model. RMS of pixel wise differences were calculated and aggregated into 15 frame bins.

---

## Neuroimaging Analysis

For each age group and model, the code to generate the events file including the feature computations, running the glm, averaging across runs and finally conducting a group lkevel analysis (merging the data into a 4D structure and then running a permutation test) is in its specific subfolders. For infants, the results are then transformed from NIHPD to MNI space for better visualization. 

Embedded within the stranet, resmem and vca models, you can also find an roi_analysis subfolder with the code to conduct the roi analsyis. The code for the creation of the ROI masks is in a seperate folder, called roi_mask.

---
