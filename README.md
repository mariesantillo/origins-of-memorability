# The Origins of Memorability in Infancy

This repository contains the full analysis pipeline for the manuscript:

**“The Origins of Memorability in Infancy”**

We investigate how computationally estimated visual memorability predicts neural responses in infants and adults during naturalistic video viewing.

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

## ResMem Memorability Model

Memorability scores were computed using the pretrained ResMem model (Bainbridge et al., 2022).

Each video was:
- Extracted into frames (25 fps)
- Scored frame-by-frame
- Averaged over 15 frames (matching TR resolution)

---

## 🧮 Neuroimaging Analysis

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
