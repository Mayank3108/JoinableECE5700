# JoinABLe – Class Project Reimplementation (ECE 570, Spring 2025)

This repository contains my reimplementation of the JoinABLe system proposed in:

**Willis et al., “JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints”**  
CVPR 2022 – [arXiv link](https://arxiv.org/abs/2111.12772)

**Original repository**: [https://github.com/AutodeskAILab/JoinABLe](https://github.com/AutodeskAILab/JoinABLe)

---

## About This Project

This codebase is part of my class project for **ECE 570: Deep Learning** at Purdue University.  
The main goal of this work was to understand the architecture proposed in the JoinABLe paper and reimplement its core components:

- The `JoinABLe` model (GNN-based joint axis prediction system).
- The `train.py` training and evaluation pipeline using PyTorch Lightning.
- Basic result visualizations based on pretrained checkpoints.

I did not attempt to train the model from scratch due to resource and time constraints. Instead, I evaluated the released pretrained checkpoints and analyzed the results.

---

## Quick Setup

> **Note**: This code builds on top of the original repository. For full functionality (e.g., pose search, notebooks), please refer to the [original JoinABLe repo](https://github.com/AutodeskAILab/JoinABLe).

### Environment Setup

Create a virtual environment and install the dependencies (tested on Python 3.7):

Dataset
Download the Fusion 360 Gallery Assembly Dataset:

- Joint Data: [j1.0.0 - 2.8 GB](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/assembly/j1.0.0/j1.0.0.7z)
- Joint Data - Preprocessed: [j1.0.0 - 514 MB](https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/assembly/j1.0.0/j1.0.0_preprocessed.7z)


Extract both into the same folder. The final path should look like:
```
data/fusion_360_joints/
  ├── joint_set_*.json
  ├── part_*.obj
  ├── joint_set_*.pickle

```

