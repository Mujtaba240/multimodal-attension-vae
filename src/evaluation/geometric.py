"""Geometric analysis of learned latent spaces."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


@torch.no_grad()
def extract_latents(model, model_name, data_loader, device, max_samples=5000):
    model.eval()
    all_mu, all_labels = [], []
    count = 0

    for images, attrs in data_loader:
        if count >= max_samples:
            break
        images, attrs = images.to(device), attrs.to(device)

        if model_name == "image_vae":
            mu, _ = model.encoder(images)
        elif model_name == "attr_vae":
            h = model.encoder(attrs)
            mu = model.fc_mu(h)
        elif model_name in ("mvae", "mmvae"):
            mu, _ = model.image_encoder(images)
        elif model_name == "attn_fuse":
            mu_img, logvar_img = model.image_encoder(images)
            mu_attr, logvar_attr = model.attr_encoder(attrs)
            mu, _, _ = model.fusion(mu_img, logvar_img, mu_attr, logvar_attr)

        all_mu.append(mu.cpu())
        all_labels.append(attrs.cpu())
        count += images.size(0)

    return torch.cat(all_mu)[:max_samples], torch.cat(all_labels)[:max_samples]


@torch.no_grad()
def extract_modality_latents(model, model_name, data_loader, device, max_samples=3000):
    """Extract latents from each modality encoder separately."""
    model.eval()
    img_mus, attr_mus = [], []
    count = 0

    for images, attrs in data_loader:
        if count >= max_samples:
            break
        images, attrs = images.to(device), attrs.to(device)

        if model_name in ("mvae", "mmvae", "attn_fuse"):
            mu_img, _ = model.image_encoder(images)
            mu_attr, _ = model.attr_encoder(attrs)
            img_mus.append(mu_img.cpu())
            attr_mus.append(mu_attr.cpu())

        count += images.size(0)

    return torch.cat(img_mus)[:max_samples], torch.cat(attr_mus)[:max_samples]


def compute_modality_gap(img_mu, attr_mu):
    img_center = img_mu.mean(dim=0)
    attr_center = attr_mu.mean(dim=0)
    gap = torch.norm(img_center - attr_center).item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        img_center.unsqueeze(0), attr_center.unsqueeze(0)
    ).item()
    return gap, cosine_sim


def compute_isotropy(mu):
    centered = mu - mu.mean(dim=0)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=1e-10)

    # Participation ratio (higher = more isotropic)
    participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    normalized_pr = participation_ratio.item() / mu.shape[1]

    # Effective dimensionality
    probs = eigenvalues / eigenvalues.sum()
    entropy = -(probs * probs.log()).sum()
    effective_dim = entropy.exp().item()

    return {
        "participation_ratio": normalized_pr,
        "effective_dim": effective_dim,
        "top5_variance_ratio": (eigenvalues[-5:].sum() / eigenvalues.sum()).item(),
    }


def plot_tsne(mu, labels, attr_idx, attr_name, save_path, title="t-SNE"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(mu.numpy()[:3000])
    attr_vals = labels[:3000, attr_idx].numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=attr_vals, cmap="coolwarm",
                         s=3, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label=attr_name)
    ax.set_title(f"{title} — colored by {attr_name}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_modality_gap(img_mu, attr_mu, save_path, title="Modality gap"):
    combined = torch.cat([img_mu[:1500], attr_mu[:1500]])
    labels = ["image"] * 1500 + ["attribute"] * 1500

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(combined.numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3B8BD4" if l == "image" else "#E8593C" for l in labels]
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, s=3, alpha=0.5)
    ax.scatter([], [], c="#3B8BD4", s=30, label="Image encoder")
    ax.scatter([], [], c="#E8593C", s=30, label="Attribute encoder")
    ax.legend()
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()