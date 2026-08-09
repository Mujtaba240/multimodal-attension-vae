"""MVAE — Multimodal VAE with Product-of-Experts fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_vae import Encoder as ImageEncoder, Decoder as ImageDecoder


class AttrEncoder(nn.Module):
    def __init__(self, num_attributes=40, latent_dim=128, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        in_dim = num_attributes
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class AttrDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_attributes=40, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], num_attributes))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def product_of_experts(mu_list, logvar_list):
    """Combine Gaussian experts via Product of Experts."""
    # Precision (inverse variance) of each expert
    precision = torch.stack([torch.exp(-lv) for lv in logvar_list])
    # Sum of precisions
    joint_precision = precision.sum(dim=0)
    joint_logvar = -torch.log(joint_precision)
    # Precision-weighted mean
    joint_mu = (torch.stack([m * p for m, p in zip(mu_list, precision)]).sum(dim=0)
                / joint_precision)
    return joint_mu, joint_logvar


class MVAE(nn.Module):
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
        mu_list, logvar_list = [], []

        # Prior as a weak expert
        batch_size = image.size(0) if image is not None else attrs.size(0)
        device = image.device if image is not None else attrs.device
        mu_prior = torch.zeros(batch_size, self.latent_dim, device=device)
        logvar_prior = torch.zeros(batch_size, self.latent_dim, device=device)
        mu_list.append(mu_prior)
        logvar_list.append(logvar_prior)

        if image is not None:
            mu_img, logvar_img = self.image_encoder(image)
            mu_list.append(mu_img)
            logvar_list.append(logvar_img)

        if attrs is not None:
            mu_attr, logvar_attr = self.attr_encoder(attrs)
            mu_list.append(mu_attr)
            logvar_list.append(logvar_attr)

        mu, logvar = product_of_experts(mu_list, logvar_list)
        z = self.reparameterize(mu, logvar)

        image_recon = self.image_decoder(z)
        attr_recon = self.attr_decoder(z)
        return image_recon, attr_recon, mu, logvar

    def loss(self, image, attrs, image_recon, attr_recon, mu, logvar, beta=1.0):
        img_recon_loss = F.mse_loss(image_recon, image, reduction="sum") / image.size(0)
        attr_recon_loss = F.binary_cross_entropy_with_logits(
            attr_recon, attrs, reduction="sum") / attrs.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / image.size(0)
        total = img_recon_loss + attr_recon_loss + beta * kl_loss
        return total, img_recon_loss, attr_recon_loss, kl_loss