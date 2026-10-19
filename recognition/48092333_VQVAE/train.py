import argparse
import json
import math
import os
import time
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DATA_ROOT, IMAGE_SHAPE, get_dataloaders, get_datasets
from modules import MODELS, load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ssim(x, y, data_range=1.0):
    x, y = x.float(), y.float()
    g = torch.exp(-(torch.arange(11, device=x.device) - 5.0) ** 2 / 4.5)
    window = (g[:, None] * g[None, :] / g.sum() ** 2)[None, None]
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = F.conv2d(x, window), F.conv2d(y, window)
    var_x = F.conv2d(x * x, window) - mu_x ** 2
    var_y = F.conv2d(y * y, window) - mu_y ** 2
    cov = F.conv2d(x * y, window) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return s.flatten(1).mean(1)


def psnr(x, y):
    mse = (x.float() - y.float()).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(1 / mse.clamp_min(1e-10))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@contextmanager
def peak_vram(stats, key="peak_vram_mb"):
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    yield
    sync()
    stats[key] = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--max-slices", type=int, help="random subset per split (smoke tests)")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--downsample", type=int, default=4)
    p.add_argument("--num-embeddings", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=64, help="also the VAE latent channels")
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--kl-warmup", type=int, default=5, help="epochs of linear KL warm-up")
    p.add_argument("--vqvae-ckpt")
    p.add_argument("--prior-dim", type=int, default=128)
    p.add_argument("--prior-layers", type=int, default=15)
    return p.parse_args()

def autoencoder_step(model, kind, x, kl_weight):
    x = x.to(DEVICE, non_blocking=True)
    if kind == "vqvae":
        recon, commitment, perplexity, _ = model(x)
        mse = F.mse_loss(recon, x)
        loss = mse + commitment
        logs = {"mse": mse.item(), "commitment": commitment.item(), "perplexity": perplexity.item()}
    else:
        recon, mu, logvar = model(x)
        sse = F.mse_loss(recon, x, reduction="sum") / len(x)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum() / len(x)
        loss = sse + kl_weight * kl
        logs = {"sse": sse.item(), "kl": kl.item()}
    with torch.no_grad():
        logs.update(ssim=ssim(recon, x).mean().item(), psnr=psnr(recon, x).mean().item())
    return loss, logs


def prior_step(prior, codes):
    codes = codes.to(DEVICE, non_blocking=True)
    loss = F.cross_entropy(prior(codes), codes)
    return loss, {"bits_per_code": loss.item() / math.log(2)}


def run_epoch(model, loader, step, epoch=None, optimiser=None):
    model.train(optimiser is not None)
    totals, n = {}, 0
    with torch.set_grad_enabled(optimiser is not None):
        for batch in loader:
            loss, logs = step(model, batch, epoch)
            if optimiser:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            logs["loss"] = loss.item()
            for k, v in logs.items():
                totals[k] = totals.get(k, 0.0) + v * len(batch)
            n += len(batch)
    return {k: v / n for k, v in totals.items()}


def plot_history(history, path):
    metrics = [k[4:] for k in history[0] if k.startswith("val_")]
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.2), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for split in ("train", "val"):
            ax.plot(epochs, [h[f"{split}_{metric}"] for h in history], label=split)
        ax.set(title=metric, xlabel="epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)