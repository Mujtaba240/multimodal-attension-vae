# Multimodal VAE: Attention-Based Fusion for Cross-Modal Generation

**Course:** Deep Learning & Applied AI (2025-26) — Sapienza University of Rome  
**Professor:** Prof. Emanuele Rodolà

## Overview

This project investigates how different multimodal fusion strategies affect the quality of cross-modal generation and learned representations in Variational Autoencoders. We compare Product-of-Experts (PoE), Mixture-of-Experts (MoE), and a proposed **Attention-Based Adaptive Fusion** mechanism, evaluating them on reconstruction metrics and through geometric analysis of the learned latent spaces.

**Dataset:** CelebA (200K face images + 40 binary attributes)

## Methods

| Model | Fusion Strategy | Description |
|-------|----------------|-------------|
| Image VAE | None (baseline) | Convolutional VAE for images only |
| Attr VAE | None (baseline) | MLP VAE for attributes only |
| MVAE | Product of Experts | Multiply modality-specific Gaussian posteriors |
| MMVAE | Mixture of Experts | Average ELBO across modality-specific experts |
| **AttnFuse** | **Attention-Based** | **Learned cross-attention over modality posteriors** |

## Results

### Reconstruction and generation quality

| Model | MSE ↓ | SSIM ↑ | FID ↓ | Attr Acc ↑ |
|-------|-------|--------|-------|-----------|
| Image VAE | 0.0080 | 0.633 | 68.8 | — |
| Attr VAE | — | — | — | 0.909 |
| MVAE | 0.0080 | 0.637 | 66.5 | 0.913 |
| MMVAE | 0.0300 | 0.443 | 166.1 | 0.937 |
| **AttnFuse** | **0.0079** | **0.637** | **64.9** | 0.914 |

### Cross-modal generation

| Model | Attr→Image MSE ↓ | Image→Attr Acc ↑ |
|-------|-------------------|------------------|
| MVAE | 0.191 | 0.813 |
| MMVAE | **0.059** | **0.877** |
| AttnFuse | 0.074 | 0.827 |

### Latent space geometry

| Model | Effective Dim ↑ | Modality Gap ↓ |
|-------|----------------|----------------|
| Image VAE | 72.5 | — |
| MVAE | 23.8 | 162.7 |
| MMVAE | 72.0 | 1.9 |
| **AttnFuse** | **76.8** | 3.9 |

### β ablation (AttnFuse)

| β | MSE ↓ | SSIM ↑ | FID ↓ | Attr Acc ↑ | Eff Dim |
|---|-------|--------|-------|-----------|---------|
| 0.1 | 0.004 | 0.746 | 42.7 | 0.890 | 118.5 |
| **1.0** | **0.008** | **0.637** | **64.9** | **0.914** | **76.8** |
| 5.0 | 0.015 | 0.524 | 114.0 | 0.855 | 24.5 |
| 10.0 | 0.059 | 0.338 | 211.8 | 0.854 | 3.1 |

## Key Findings

- AttnFuse achieves the best reconstruction quality (FID 64.9) and highest latent space isotropy (effective dim 76.8) among all multimodal models
- PoE (MVAE) severely collapses the latent space (23.8 effective dims) despite competitive reconstruction
- MoE (MMVAE) preserves latent structure but sacrifices reconstruction quality (FID 166.1)
- AttnFuse degrades less under missing modalities compared to MVAE
- β=1.0 provides the optimal balance between reconstruction and latent space structure

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