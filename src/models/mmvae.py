"""MMVAE — Multimodal VAE with Mixture-of-Experts fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_vae import Encoder as ImageEncoder, Decoder as ImageDecoder
from .mvae import AttrEncoder, AttrDecoder


class MMVAE(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3, num_attributes=40,
                 encoder_hidden=[32, 64, 128, 256], decoder_hidden=[256, 128, 64, 32],
                 attr_hidden=[128, 128]):
        super().__init__()
        self.latent_dim = latent_dim

        self.image_encoder = ImageEncoder(latent_dim, image_channels, encoder_hidden)
        self.image_decoder = ImageDecoder(latent_dim, image_channels, decoder_hidden)
        self.attr_encoder = AttrEncoder(num_attributes, latent_dim, attr_hidden)
        self.attr_decoder = AttrDecoder(latent_dim, num_attributes, attr_hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image=None, attrs=None):
        # Encode from each available modality
        mu_img, logvar_img = self.image_encoder(image)
        mu_attr, logvar_attr = self.attr_encoder(attrs)

        # Sample z from each expert independently
        z_img = self.reparameterize(mu_img, logvar_img)
        z_attr = self.reparameterize(mu_attr, logvar_attr)

        # Decode from both latents (cross-generation)
        image_recon_from_img = self.image_decoder(z_img)
        attr_recon_from_img = self.attr_decoder(z_img)
        image_recon_from_attr = self.image_decoder(z_attr)
        attr_recon_from_attr = self.attr_decoder(z_attr)

        return {
            "z_img": z_img, "z_attr": z_attr,
            "mu_img": mu_img, "logvar_img": logvar_img,
            "mu_attr": mu_attr, "logvar_attr": logvar_attr,
            "image_recon_from_img": image_recon_from_img,
            "attr_recon_from_img": attr_recon_from_img,
            "image_recon_from_attr": image_recon_from_attr,
            "attr_recon_from_attr": attr_recon_from_attr,
        }

    def loss(self, image, attrs, out, beta=1.0):
        # Image expert ELBO
        img_recon_img = F.mse_loss(out["image_recon_from_img"], image, reduction="sum") / image.size(0)
        img_recon_attr = F.binary_cross_entropy_with_logits(
            out["attr_recon_from_img"], attrs, reduction="sum") / image.size(0)
        kl_img = -0.5 * torch.sum(
            1 + out["logvar_img"] - out["mu_img"].pow(2) - out["logvar_img"].exp()) / image.size(0)
        elbo_img = img_recon_img + img_recon_attr + beta * kl_img

        # Attr expert ELBO
        attr_recon_img = F.mse_loss(out["image_recon_from_attr"], image, reduction="sum") / image.size(0)
        attr_recon_attr = F.binary_cross_entropy_with_logits(
            out["attr_recon_from_attr"], attrs, reduction="sum") / image.size(0)
        kl_attr = -0.5 * torch.sum(
            1 + out["logvar_attr"] - out["mu_attr"].pow(2) - out["logvar_attr"].exp()) / image.size(0)
        elbo_attr = attr_recon_img + attr_recon_attr + beta * kl_attr

        # Mixture: average of expert ELBOs
        total = 0.5 * (elbo_img + elbo_attr)
        return total, elbo_img, elbo_attr