# Multimodal VAE: Attention-Based Fusion for Cross-Modal Generation

**Course:** Deep Learning & Applied AI (2025-26) — Sapienza University of Rome  
**Professor:** Prof. Emanuele Rodolà

## Overview

This project investigates how different multimodal fusion strategies affect the quality of cross-modal generation and learned representations in Variational Autoencoders. We compare Product-of-Experts (PoE), Mixture-of-Experts (MoE), and a proposed **Attention-Based Adaptive Fusion** mechanism, evaluating them on reconstruction metrics and through geometric analysis of the learned latent spaces.

**Dataset:** CelebA (200K face images + 40 binary attributes)

## Project Structure

├── src/
│ ├── models/ # VAE architectures
│ ├── data/ # Dataset loading and preprocessing
│ ├── utils/ # Training utilities, checkpointing, paths
│ └── evaluation/ # Metrics and geometric analysis
├── configs/ # Hyperparameter configs
├── notebooks/ # Colab training notebooks
├── results/
│ ├── figures/
│ ├── tables/
│ └── checkpoints/
├── report/ # LaTeX source for the course report
└── scripts/ # Training entry points

## Methods

| Model | Fusion Strategy | Description |
|-------|----------------|-------------|
| Unimodal VAE | None (baseline) | Separate image and attribute VAEs |
| MVAE | Product of Experts | Multiply modality-specific Gaussian posteriors |
| MMVAE | Mixture of Experts | Average ELBO across modality-specific experts |
| AttnFuse | Attention-Based | Learned cross-attention over modality posteriors |

## Setup

```bash
pip install -r requirements.txt
```

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes.
- Wu & Goodman (2018). Multimodal Generative Models for Scalable Weakly-Supervised Learning.
- Shi et al. (2019). Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models.
- Shi et al. (2021). Relating by Contrasting: A Data-efficient Framework for Multimodal Generative Models.
- Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
- Vaswani et al. (2017). Attention Is All You Need.
- Liu et al. (2015). Deep Learning Face Attributes in the Wild (CelebA).
- Sutter et al. (2021). Generalized Multimodal ELBO.
- Daunhawer et al. (2022). On the Limitations of Multimodal VAEs.
- Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID).