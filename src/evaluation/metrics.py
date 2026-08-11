"""Evaluation metrics for all VAE models."""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torchvision.models import inception_v3
from torchmetrics.image import StructuralSimilarityIndexMeasure


class InceptionFeatureExtractor:
    def __init__(self, device):
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def extract(self, images):
        # Resize to 299x299 for Inception
        x = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def compute_fid(real_features, gen_features):
    mu1, sigma1 = real_features.mean(0).cpu().numpy(), np.cov(real_features.cpu().numpy(), rowvar=False)
    mu2, sigma2 = gen_features.mean(0).cpu().numpy(), np.cov(gen_features.cpu().numpy(), rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


class VAEMetrics:
    def __init__(self, device="cuda"):
        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.inception = None

    def _get_inception(self):
        if self.inception is None:
            self.inception = InceptionFeatureExtractor(self.device)
        return self.inception

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


@torch.no_grad()
def evaluate_model(model, model_name, test_loader, device, num_fid_samples=5000):
    model.eval()
    metrics = VAEMetrics(device)
    inception = metrics._get_inception()

    all_mse, all_ssim = [], []
    all_attr_acc = []
    real_features, gen_features = [], []
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
                real_features.append(inception.extract(images))
                gen_features.append(inception.extract(recon))
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

    if real_features and gen_features:
        real_features = torch.cat(real_features)[:num_fid_samples]
        gen_features = torch.cat(gen_features)[:num_fid_samples]
        results["fid"] = compute_fid(real_features, gen_features)

    return results


@torch.no_grad()
def evaluate_cross_modal(model, model_name, test_loader, device):
    model.eval()
    metrics = VAEMetrics(device)

    img_from_attr_mse, attr_from_img_acc = [], []

    for images, attrs in test_loader:
        images, attrs = images.to(device), attrs.to(device)

        if model_name == "mvae":
            img_recon, _, _, _ = model(image=None, attrs=attrs)
            img_from_attr_mse.append(F.mse_loss(img_recon, images).item())

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