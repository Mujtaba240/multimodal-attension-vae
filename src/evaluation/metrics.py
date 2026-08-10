"""Evaluation metrics for all VAE models."""

import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


class VAEMetrics:
    def __init__(self, device="cuda"):
        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def reconstruction_mse(self, original, reconstructed):
        return F.mse_loss(reconstructed, original).item()

    def reconstruction_ssim(self, original, reconstructed):
        self.ssim.reset()
        return self.ssim(reconstructed, original).item()

    def attribute_accuracy(self, true_attrs, predicted_logits):
        preds = (torch.sigmoid(predicted_logits) > 0.5).float()
        acc = (preds == true_attrs).float().mean().item()
        per_attr_acc = (preds == true_attrs).float().mean(dim=0)
        return acc, per_attr_acc

    def compute_fid(self, real_images, generated_images, batch_size=64):
        self.fid.reset()
        for i in range(0, len(real_images), batch_size):
            batch_real = real_images[i:i+batch_size].to(self.device)
            batch_gen = generated_images[i:i+batch_size].to(self.device)
            self.fid.update(batch_real, real=True)
            self.fid.update(batch_gen, real=False)
        return self.fid.compute().item()


@torch.no_grad()
def evaluate_model(model, model_name, test_loader, device, num_fid_samples=5000):
    model.eval()
    metrics = VAEMetrics(device)

    all_mse, all_ssim = [], []
    all_attr_acc = []
    real_images, gen_images = [], []
    count = 0

    for images, attrs in test_loader:
        images, attrs = images.to(device), attrs.to(device)

        if model_name == "image_vae":
            recon, mu, logvar = model(images)
            attr_recon = None
        elif model_name == "attr_vae":
            attr_recon, mu, logvar = model(attrs)
            recon = None
        elif model_name in ("mvae", "attn_fuse"):
            if model_name == "attn_fuse":
                recon, attr_recon, mu, logvar, _ = model(images, attrs)
            else:
                recon, attr_recon, mu, logvar = model(images, attrs)
        elif model_name == "mmvae":
            out = model(images, attrs)
            recon = out["image_recon_from_img"]
            attr_recon = out["attr_recon_from_attr"]

        if recon is not None:
            all_mse.append(metrics.reconstruction_mse(images, recon))
            all_ssim.append(metrics.reconstruction_ssim(images, recon))

            if count < num_fid_samples:
                real_images.append(images.cpu())
                gen_images.append(recon.cpu())
                count += images.size(0)

        if attr_recon is not None:
            acc, _ = metrics.attribute_accuracy(attrs, attr_recon)
            all_attr_acc.append(acc)

    results = {}
    if all_mse:
        results["mse"] = np.mean(all_mse)
        results["ssim"] = np.mean(all_ssim)
    if all_attr_acc:
        results["attr_accuracy"] = np.mean(all_attr_acc)

    if real_images and gen_images:
        real_images = torch.cat(real_images)[:num_fid_samples]
        gen_images = torch.cat(gen_images)[:num_fid_samples]
        results["fid"] = metrics.compute_fid(real_images, gen_images)

    return results


@torch.no_grad()
def evaluate_cross_modal(model, model_name, test_loader, device):
    """Generate images from attributes only, and attributes from images only."""
    model.eval()
    metrics = VAEMetrics(device)

    img_from_attr_mse, attr_from_img_acc = [], []

    for images, attrs in test_loader:
        images, attrs = images.to(device), attrs.to(device)

        if model_name == "mvae":
            # Image from attributes only
            img_recon, _, _, _ = model(image=None, attrs=attrs)
            img_from_attr_mse.append(F.mse_loss(img_recon, images).item())

            # Attributes from image only
            _, attr_recon, _, _ = model(image=images, attrs=None)
            acc, _ = metrics.attribute_accuracy(attrs, attr_recon)
            attr_from_img_acc.append(acc)

        elif model_name == "mmvae":
            out = model(images, attrs)
            img_from_attr_mse.append(F.mse_loss(out["image_recon_from_attr"], images).item())
            acc, _ = metrics.attribute_accuracy(attrs, out["attr_recon_from_img"])
            attr_from_img_acc.append(acc)

        elif model_name == "attn_fuse":
            img_recon, _, _, _, _ = model(image=None, attrs=attrs)
            img_from_attr_mse.append(F.mse_loss(img_recon, images).item())

            _, attr_recon, _, _, _ = model(image=images, attrs=None)
            acc, _ = metrics.attribute_accuracy(attrs, attr_recon)
            attr_from_img_acc.append(acc)

    return {
        "cross_img_mse": np.mean(img_from_attr_mse) if img_from_attr_mse else None,
        "cross_attr_acc": np.mean(attr_from_img_acc) if attr_from_img_acc else None,
    }