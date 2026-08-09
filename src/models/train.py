"""Training script for VAE models."""

import os
import sys
import yaml
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataloaders
from src.models import ImageVAE, AttributeVAE
from src.utils import set_seed, get_device, save_checkpoint, load_checkpoint, find_latest_checkpoint


def get_beta(epoch, warmup_epochs, target_beta):
    if epoch < warmup_epochs:
        return target_beta * (epoch + 1) / warmup_epochs
    return target_beta


def train_image_vae(config, data_dir, ckpt_dir):
    device = get_device()
    set_seed(config["seed"])

    train_loader, val_loader, _ = get_dataloaders(
        data_dir,
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        crop_size=config["data"]["crop_size"],
        num_workers=config["data"]["num_workers"],
    )

    model = ImageVAE(
        latent_dim=config["model"]["latent_dim"],
        channels=config["model"]["image_channels"],
        encoder_hidden=config["model"]["encoder_hidden"],
        decoder_hidden=config["model"]["decoder_hidden"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    save_dir = os.path.join(ckpt_dir, "image_vae")
    start_epoch = 0

    latest = find_latest_checkpoint(save_dir, prefix="ImageVAE")
    if latest:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, latest, device)
        start_epoch += 1

    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        beta = get_beta(epoch, config["loss"]["beta_warmup_epochs"], config["loss"]["beta"])
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for images, _ in pbar:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = model.loss(images, recon, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}", recon=f"{recon_loss.item():.2f}",
                             kl=f"{kl_loss.item():.2f}", beta=f"{beta:.3f}")

        scheduler.step()
        n = len(train_loader)
        print(f"[Epoch {epoch+1}] loss={epoch_loss/n:.2f} recon={epoch_recon/n:.2f} kl={epoch_kl/n:.2f}")

        if (epoch + 1) % config["evaluation"]["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss / n, save_dir)


def train_attr_vae(config, data_dir, ckpt_dir):
    device = get_device()
    set_seed(config["seed"])

    train_loader, val_loader, _ = get_dataloaders(
        data_dir,
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        crop_size=config["data"]["crop_size"],
        num_workers=config["data"]["num_workers"],
    )

    model = AttributeVAE(
        num_attributes=config["model"]["num_attributes"],
        latent_dim=config["model"]["latent_dim"],
        hidden_dims=config["model"]["attr_hidden"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    save_dir = os.path.join(ckpt_dir, "attr_vae")
    start_epoch = 0

    latest = find_latest_checkpoint(save_dir, prefix="AttributeVAE")
    if latest:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, latest, device)
        start_epoch += 1

    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        beta = get_beta(epoch, config["loss"]["beta_warmup_epochs"], config["loss"]["beta"])
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for _, attrs in pbar:
            attrs = attrs.to(device)
            recon, mu, logvar = model(attrs)
            loss, recon_loss, kl_loss = model.loss(attrs, recon, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}", beta=f"{beta:.3f}")

        scheduler.step()
        n = len(train_loader)
        print(f"[Epoch {epoch+1}] loss={epoch_loss/n:.2f} recon={epoch_recon/n:.2f} kl={epoch_kl/n:.2f}")

        if (epoch + 1) % config["evaluation"]["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, epoch_loss / n, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["image_vae", "attr_vae"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.model == "image_vae":
        train_image_vae(config, args.data_dir, args.ckpt_dir)
    elif args.model == "attr_vae":
        train_attr_vae(config, args.data_dir, args.ckpt_dir)