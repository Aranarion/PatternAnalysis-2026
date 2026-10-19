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
