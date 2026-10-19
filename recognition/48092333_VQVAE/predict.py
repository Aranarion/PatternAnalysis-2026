import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

from dataset import get_datasets, patient_id
from modules import load_model
from train import DEVICE, count_parameters, peak_vram, psnr, ssim, sync


def summarise(values):
    return {"mean": values.mean().item(), "std": values.std().item(),
            "min": values.min().item(), "max": values.max().item()}


@torch.no_grad()
def batched(fn, inputs, batch_size=64):
    return torch.cat([fn(inputs[i:i + batch_size].to(DEVICE)).cpu() for i in range(0, len(inputs), batch_size)])


def timed(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, time.perf_counter() - start


@torch.no_grad()
def time_ms(fn, repeats=20):
    for _ in range(3):
        fn()
    return 1000 * timed(lambda: [fn() for _ in range(repeats)])[1] / repeats


@torch.no_grad()
def nearest_ssim(queries, references, k=5, chunk=256):
    """Highest SSIM between each query and its k nearest references by pixel L2.

    Returns (best SSIM, index of that reference) per query.
    """
    refs = references.flatten(1).float().to(DEVICE)
    best, best_idx = [], []
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk].to(DEVICE).float()
        candidates = torch.cdist(q.flatten(1), refs).topk(k, largest=False).indices.cpu()
        scores = ssim(q.repeat_interleave(k, 0), references[candidates.flatten()].to(DEVICE)).view(-1, k)
        values, j = scores.max(1)
        best.append(values.cpu())
        best_idx.append(candidates.gather(1, j.cpu()[:, None])[:, 0])
    return torch.cat(best), torch.cat(best_idx)


@torch.no_grad()
def pairwise_ssim(images, n_pairs=1000, seed=0):
    """Mean SSIM between random pairs; values near 1 indicate mode collapse."""
    a, b = torch.randint(0, len(images), (2, n_pairs), generator=torch.Generator().manual_seed(seed))
    a, b = a[a != b], b[a != b]
    return ssim(images[a].to(DEVICE), images[b].to(DEVICE)).mean().item()

class InceptionFeatures(nn.Module):
    """2048-dimenionsal torchvision Inception-v3 features (FID comparable within this project only)."""

    def __init__(self):
        super().__init__()
        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Identity()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()

    @torch.no_grad()
    def forward(self, images, batch_size=64):
        feats = []
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size].to(DEVICE).float().repeat(1, 3, 1, 1)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            feats.append(self.net((x - self.mean) / self.std).cpu())
        return torch.cat(feats).double()


def frechet_distance(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = torch.cov(f1.t()), torch.cov(f2.t())
    sqrt_trace = torch.linalg.eigvals(s1 @ s2).real.clamp_min(0).sqrt().sum()
    return ((mu1 - mu2).pow(2).sum() + s1.trace() + s2.trace() - 2 * sqrt_trace).item()


def density_coverage(real, fake, k=5):
    """Density and coverage of `fake` relative to `real` (Naeem et al. 2020)."""
    real, fake = real.float(), fake.float()
    radii = torch.cdist(real, real).topk(k + 1, largest=False).values[:, -1]
    inside = torch.cdist(real, fake) < radii[:, None]
    return {"density": inside.sum().item() / (k * len(fake)),
            "coverage": inside.any(1).float().mean().item()}


def generation_stats(nn_train, images, threshold):
    return {"nn_ssim_train": summarise(nn_train),
            "frac_above_0.6": (nn_train > 0.6).float().mean().item(),
            "frac_possible_copy": (nn_train > threshold).float().mean().item(),
            "frac_near_duplicate": (nn_train > 0.95).float().mean().item(),
            "pairwise_ssim": pairwise_ssim(images)}


def image_grid(rows, titles, path, title):
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(1.5 * len(rows[0]), 2.9 * len(rows)),
                             squeeze=False)
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            axes[r, c].imshow(image.squeeze().float().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(titles[r][c], fontsize=7)
            axes[r, c].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
