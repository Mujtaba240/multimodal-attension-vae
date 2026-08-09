"""MLP-based VAE for CelebA binary attributes (40-dim)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeVAE(nn.Module):
    def __init__(self, num_attributes=40, latent_dim=128, hidden_dims=[128, 128]):
        super().__init__()
        self.latent_dim = latent_dim

        enc_layers = []
        in_dim = num_attributes
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h_dim))
            enc_layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h_dim))
            dec_layers.append(nn.ReLU())
            in_dim = h_dim
        dec_layers.append(nn.Linear(hidden_dims[0], num_attributes))
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, attrs):
        h = self.encoder(attrs)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        logits = self.decoder(z)
        return torch.sigmoid(logits)

    def loss(self, attrs, recon, mu, logvar, beta=1.0):
        recon_loss = F.binary_cross_entropy_with_logits(recon, attrs, reduction="sum") / attrs.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / attrs.size(0)
        return recon_loss + beta * kl_loss, recon_loss, kl_loss