# The Origins of Memorability in Infancy

This repository contains the full analysis pipeline for the manuscript:

**“The Origins of Memorability in Infancy”**, investigating how memorability estimated by deep neural networks such as ResMem (Needell & Bainbridge, 2022) predict neural responses in two month old infants and adults during awake naturalistic video fMRI.

---

##  Overview

This project tests whether frame-level memorability scores derived from the ResMem model predict brain activity during fMRI scanning.

The pipeline includes:

1. Frame extraction from videos (.mp4)
2. Memorability prediction using ResMem (Needell et al. 2022)
3. Parametric modulator event-table generation
4. First-level GLM analyses
5. Second-level permutation testing (TFCE)
6. Transformation to MNI space
7. Cluster peak extraction

---

## Feature Computation 
All the code to calculate the different features can be found in memorability/feature_computation/ with one subfolder for each feature.
### ResMem Memorability Model
Memorability scores were computed using the pretrained ResMem model (Bainbridge et al., 2022).
Each video was:
- Extracted into frames (25 fps)
- Scored frame-by-frame
- Averaged over 15 frames to match with 0.61 TR resolution
### Entropy Calculation 
Visual entropy scores were computed with VCA (Manon et al.) using 32x32 block sizes and scores were averaged every 15 frames. 
### Saliency calculation 
Saliency predictions were generated using STRA-Net. Optic flow images were computed with RAFT and fed into the STRANET model. RMS of pixel wise differences were calculated and aggregated into 15 frame bins.

---

## Neuroimaging Analysis

### First-Level Analysis
Implemented using:

- Nilearn
- Nibabel
- Custom parametric modulators
- Motion confound regression

### Second-Level Analysis
Performed using:

- FSL `randomise`
- TFCE correction
- One-sample permutation testing

### Spatial Normalization
Maps were transformed to MNI space using ANTs.

---
