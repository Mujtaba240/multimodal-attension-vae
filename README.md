# Multimodal VAE: Attention-Based Fusion for Cross-Modal Generation

**Course:** Deep Learning & Applied AI (2025-26) — Sapienza University of Rome  
**Professor:** Prof. Emanuele Rodolà

## Overview

This project investigates how different multimodal fusion strategies affect cross-modal generation quality and latent space geometry in Variational Autoencoders. Using the CelebA dataset (face images + binary attributes), we compare Product-of-Experts, Mixture-of-Experts, and a proposed attention-based adaptive fusion mechanism.

## Project Structure
├── src/
│ ├── models/ # ImageVAE, AttrVAE, MVAE, MMVAE, AttnFuse
│ ├── data/ # CelebA dataloader
│ ├── utils/ # Checkpointing, paths, reproducibility
│ └── evaluation/ # Metrics, visualization, geometric analysis
├── configs/ # Training hyperparameters
├── scripts/ # Training and data download scripts
├── notebooks/ # Colab launcher
└── report/ # LaTeX report source

## Methods

| Model | Fusion Strategy |
|-------|----------------|
| Image VAE / Attr VAE | Unimodal baselines |
| MVAE | Product of Experts |
| MMVAE | Mixture of Experts |
| **AttnFuse (ours)** | **Attention-based adaptive fusion** |

## Setup

```bash
pip install -r requirements.txt
```

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes.
- Wu & Goodman (2018). Multimodal Generative Models for Scalable Weakly-Supervised Learning.
- Shi et al. (2019). Variational Mixture-of-Experts Autoencoders.
- Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
- Vaswani et al. (2017). Attention Is All You Need.
- Liu et al. (2015). Deep Learning Face Attributes in the Wild.