"""Convolutional VAE for CelebA images (64x64x3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim, channels=3, hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        layers = []
        in_ch = channels
        for h_dim in hidden_dims:
            layers.append(nn.Conv2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(h_dim))
            layers.append(nn.ReLU())
            in_ch = h_dim
        self.conv = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, channels=3, hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * 4 * 4)
        self.init_dim = hidden_dims[0]

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                             kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(hidden_dims[-1], channels,
                                         kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.init_dim, 4, 4)
        return self.deconv(h)


class ImageVAE(nn.Module):
    def __init__(self, latent_dim=128, channels=3,
                 encoder_hidden=[32, 64, 128, 256],
                 decoder_hidden=[256, 128, 64, 32]):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, channels, encoder_hidden)
        self.decoder = Decoder(latent_dim, channels, decoder_hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)

    def loss(self, x, recon, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kl_loss, recon_loss, kl_loss