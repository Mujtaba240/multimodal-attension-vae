"""Checkpoint save/load — persists to Google Drive across Colab sessions."""

import os
import torch


def save_checkpoint(model, optimizer, epoch, loss, path, filename=None):
    if filename is None:
        filename = f"{model.__class__.__name__}_epoch{epoch:03d}.pt"
    filepath = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(model, optimizer, filepath, device="cuda"):
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss: {ckpt['loss']:.4f})")
    return model, optimizer, ckpt["epoch"], ckpt["loss"]


def find_latest_checkpoint(path, prefix=None):
    if not os.path.exists(path):
        return None
    files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    return os.path.join(path, files[-1]) if files else None
