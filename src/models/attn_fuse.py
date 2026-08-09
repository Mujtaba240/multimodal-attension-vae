"""AttnFuse — Multimodal VAE with attention-based adaptive fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_vae import Encoder as ImageEncoder, Decoder as ImageDecoder
from .mvae import AttrEncoder, AttrDecoder


class AttentionFusion(nn.Module):
    """Learns to dynamically weight modality-specific posteriors."""
    def __init__(self, latent_dim):
        super().__init__()
        # Project each (mu, logvar) pair to a key for attention
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key_img = nn.Linear(latent_dim * 2, latent_dim)
        self.key_attr = nn.Linear(latent_dim * 2, latent_dim)
        self.scale = latent_dim ** 0.5

    def forward(self, mu_img, logvar_img, mu_attr, logvar_attr):
        # Global query from average of means
        q = self.query(0.5 * (mu_img + mu_attr))

        # Keys from concatenated (mu, logvar) of each modality
        k_img = self.key_img(torch.cat([mu_img, logvar_img], dim=-1))
        k_attr = self.key_attr(torch.cat([mu_attr, logvar_attr], dim=-1))

        # Attention scores
        keys = torch.stack([k_img, k_attr], dim=1)
        scores = torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1) / self.scale
        weights = F.softmax(scores, dim=1)

        # Weighted combination of posteriors
        w_img = weights[:, 0:1]
        w_attr = weights[:, 1:2]

        fused_mu = w_img * mu_img + w_attr * mu_attr
        fused_logvar = torch.log(w_img.pow(2) * logvar_img.exp() + w_attr.pow(2) * logvar_attr.exp())

        return fused_mu, fused_logvar, weights


class AttnFuseVAE(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3, num_attributes=40,
                 encoder_hidden=[32, 64, 128, 256], decoder_hidden=[256, 128, 64, 32],
                 attr_hidden=[128, 128]):
        super().__init__()
        self.latent_dim = latent_dim

        self.image_encoder = ImageEncoder(latent_dim, image_channels, encoder_hidden)
        self.image_decoder = ImageDecoder(latent_dim, image_channels, decoder_hidden)
        self.attr_encoder = AttrEncoder(num_attributes, latent_dim, attr_hidden)
        self.attr_decoder = AttrDecoder(latent_dim, num_attributes, attr_hidden)
        self.fusion = AttentionFusion(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image=None, attrs=None):
        has_image = image is not None
        has_attrs = attrs is not None

        if has_image and has_attrs:
            mu_img, logvar_img = self.image_encoder(image)
            mu_attr, logvar_attr = self.attr_encoder(attrs)
            mu, logvar, weights = self.fusion(mu_img, logvar_img, mu_attr, logvar_attr)
        elif has_image:
            mu, logvar = self.image_encoder(image)
            weights = None
        elif has_attrs:
            mu, logvar = self.attr_encoder(attrs)
            weights = None

        z = self.reparameterize(mu, logvar)
        image_recon = self.image_decoder(z)
        attr_recon = self.attr_decoder(z)
        return image_recon, attr_recon, mu, logvar, weights

    def loss(self, image, attrs, image_recon, attr_recon, mu, logvar, beta=1.0):
        img_recon_loss = F.mse_loss(image_recon, image, reduction="sum") / image.size(0)
        attr_recon_loss = F.binary_cross_entropy_with_logits(
            attr_recon, attrs, reduction="sum") / attrs.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / image.size(0)
        total = img_recon_loss + attr_recon_loss + beta * kl_loss
        return total, img_recon_loss, attr_recon_loss, kl_loss