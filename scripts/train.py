"""Training script for all VAE models."""

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
from src.models import ImageVAE, AttributeVAE, MVAE, MMVAE, AttnFuseVAE
from src.utils import set_seed, get_device, save_checkpoint, load_checkpoint, find_latest_checkpoint


def get_beta(epoch, warmup_epochs, target_beta):
    if epoch < warmup_epochs:
        return target_beta * (epoch + 1) / warmup_epochs
    return target_beta


def train_one_epoch(model, model_name, train_loader, optimizer, device, beta, grad_clip, epoch, total_epochs):
    model.train()
    epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
    extra_metrics = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for images, attrs in pbar:
        images, attrs = images.to(device), attrs.to(device)

        if model_name == "image_vae":
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = model.loss(images, recon, mu, logvar, beta=beta)

        elif model_name == "attr_vae":
            recon, mu, logvar = model(attrs)
            loss, recon_loss, kl_loss = model.loss(attrs, recon, mu, logvar, beta=beta)

        elif model_name == "mvae":
            img_recon, attr_recon, mu, logvar = model(images, attrs)
            loss, img_rl, attr_rl, kl_loss = model.loss(images, attrs, img_recon, attr_recon, mu, logvar, beta=beta)
            recon_loss = img_rl + attr_rl

        elif model_name == "mmvae":
            out = model(images, attrs)
            loss, elbo_img, elbo_attr = model.loss(images, attrs, out, beta=beta)
            recon_loss = loss
            kl_loss = torch.tensor(0.0)

        elif model_name == "attn_fuse":
            img_recon, attr_recon, mu, logvar, weights = model(images, attrs)
            loss, img_rl, attr_rl, kl_loss = model.loss(images, attrs, img_recon, attr_recon, mu, logvar, beta=beta)
            recon_loss = img_rl + attr_rl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()
        pbar.set_postfix(loss=f"{loss.item():.2f}", beta=f"{beta:.3f}")

    n = len(train_loader)
    return epoch_loss / n, epoch_recon / n, epoch_kl / n


def build_model(model_name, config):
    cfg = config["model"]
    if model_name == "image_vae":
        return ImageVAE(cfg["latent_dim"], cfg["image_channels"],
                        cfg["encoder_hidden"], cfg["decoder_hidden"])
    elif model_name == "attr_vae":
        return AttributeVAE(cfg["num_attributes"], cfg["latent_dim"], cfg["attr_hidden"])
    elif model_name == "mvae":
        return MVAE(cfg["latent_dim"], cfg["image_channels"], cfg["num_attributes"],
                    cfg["encoder_hidden"], cfg["decoder_hidden"], cfg["attr_hidden"])
    elif model_name == "mmvae":
        return MMVAE(cfg["latent_dim"], cfg["image_channels"], cfg["num_attributes"],
                     cfg["encoder_hidden"], cfg["decoder_hidden"], cfg["attr_hidden"])
    elif model_name == "attn_fuse":
        return AttnFuseVAE(cfg["latent_dim"], cfg["image_channels"], cfg["num_attributes"],
                           cfg["encoder_hidden"], cfg["decoder_hidden"], cfg["attr_hidden"])


def train(config, model_name, data_dir, ckpt_dir):
    device = get_device()
    set_seed(config["seed"])

    train_loader, val_loader, _ = get_dataloaders(
        data_dir,
        batch_size=config["training"]["batch_size"],
        image_size=config["data"]["image_size"],
        crop_size=config["data"]["crop_size"],
        num_workers=config["data"]["num_workers"],
    )

    model = build_model(model_name, config).to(device)
    optimizer = Adam(model.parameters(), lr=config["training"]["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    save_dir = os.path.join(ckpt_dir, model_name)
    start_epoch = 0

    # Resume from checkpoint
    class_name = model.__class__.__name__
    latest = find_latest_checkpoint(save_dir, prefix=class_name)
    if latest:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, latest, device)
        start_epoch += 1

    total_epochs = config["training"]["epochs"]
    for epoch in range(start_epoch, total_epochs):
        beta = get_beta(epoch, config["loss"]["beta_warmup_epochs"], config["loss"]["beta"])

        avg_loss, avg_recon, avg_kl = train_one_epoch(
            model, model_name, train_loader, optimizer, device,
            beta, config["training"]["grad_clip"], epoch, total_epochs,
        )
        scheduler.step()

        print(f"[Epoch {epoch+1}] loss={avg_loss:.2f} recon={avg_recon:.2f} kl={avg_kl:.2f}")

        if (epoch + 1) % config["evaluation"]["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, save_dir)

    # Always save final checkpoint
    save_checkpoint(model, optimizer, total_epochs - 1, avg_loss, save_dir, filename=f"{class_name}_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["image_vae", "attr_vae", "mvae", "mmvae", "attn_fuse"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.model, args.data_dir, args.ckpt_dir)