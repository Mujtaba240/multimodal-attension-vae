"""Visualization utilities for VAE evaluation."""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os


@torch.no_grad()
def plot_reconstructions(model, model_name, test_loader, device, save_path, n=8):
    model.eval()
    images, attrs = next(iter(test_loader))
    images, attrs = images.to(device), attrs.to(device)

    if model_name == "image_vae":
        recon, _, _ = model(images)
    elif model_name in ("mvae",):
        recon, _, _, _ = model(images, attrs)
    elif model_name == "mmvae":
        out = model(images, attrs)
        recon = out["image_recon_from_img"]
    elif model_name == "attn_fuse":
        recon, _, _, _, _ = model(images, attrs)

    images = images[:n].cpu()
    recon = recon[:n].cpu()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0][i].imshow(images[i].permute(1, 2, 0).clamp(0, 1))
        axes[0][i].axis("off")
        axes[1][i].imshow(recon[i].permute(1, 2, 0).clamp(0, 1))
        axes[1][i].axis("off")
    axes[0][0].set_title("Original", fontsize=10)
    axes[1][0].set_title("Reconstructed", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def plot_samples(model, model_name, device, save_path, n=16):
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)

    if model_name == "image_vae":
        samples = model.decoder(z)
    elif model_name in ("mvae", "mmvae", "attn_fuse"):
        samples = model.image_decoder(z)

    samples = samples.cpu()
    rows = n // 8

    fig, axes = plt.subplots(rows, 8, figsize=(16, 2 * rows))
    if rows == 1:
        axes = axes.unsqueeze(0)
    for i in range(n):
        axes[i // 8][i % 8].imshow(samples[i].permute(1, 2, 0).clamp(0, 1))
        axes[i // 8][i % 8].axis("off")
    plt.suptitle(f"Samples: {model_name}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def plot_interpolation(model, model_name, test_loader, device, save_path, steps=10):
    model.eval()
    images, attrs = next(iter(test_loader))
    images, attrs = images[:2].to(device), attrs[:2].to(device)

    if model_name == "image_vae":
        mu1, _ = model.encoder(images[0:1])
        mu2, _ = model.encoder(images[1:2])
    elif model_name in ("mvae", "attn_fuse"):
        mu1, _ = model.image_encoder(images[0:1])
        mu2, _ = model.image_encoder(images[1:2])
    elif model_name == "mmvae":
        mu1, _ = model.image_encoder(images[0:1])
        mu2, _ = model.image_encoder(images[1:2])

    alphas = torch.linspace(0, 1, steps, device=device)
    interpolations = []
    for alpha in alphas:
        z = (1 - alpha) * mu1 + alpha * mu2
        if model_name == "image_vae":
            img = model.decoder(z)
        else:
            img = model.image_decoder(z)
        interpolations.append(img.cpu())

    interpolations = torch.cat(interpolations)

    fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2))
    for i in range(steps):
        axes[i].imshow(interpolations[i].permute(1, 2, 0).clamp(0, 1))
        axes[i].axis("off")
    plt.suptitle(f"Latent interpolation: {model_name}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()