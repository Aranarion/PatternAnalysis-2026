"""Train the VAE baseline, the VQ-VAE, or the PixelCNN prior over VQ-VAE codes.

    python train.py --model vae
    python train.py --model vqvae
    python train.py --model prior --vqvae-ckpt outputs/vqvae/best.pt

Writes best.pt (lowest validation loss), history.json and curves.png to <out-dir>/<model>/.
"""

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
    """Per-image SSIM (Wang et al. 2004)"""
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
    p.add_argument("--save-epochs", type=int, nargs="*", default=[], help="also save checkpoints at these epochs")
    p.add_argument("--prior-dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW decoupled weight decay")
    p.add_argument("--patience", type=int, default=0, help="stop after this many epochs without improvement (0 = off)")
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


def fit(model, config, loaders, step, args, extra):
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "best.pt")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs)
    history, best, best_epoch, start = [], float("inf"), 0, time.time()

    for epoch in range(1, args.epochs + 1):
        stats, t0 = {}, time.time()
        with peak_vram(stats):
            train = run_epoch(model, loaders["train"], step, epoch, optimiser)
        scheduler.step()
        val = run_epoch(model, loaders["validate"], step)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                  **{f"val_{k}": v for k, v in val.items()}, "epoch_s": time.time() - t0, **stats}
        history.append(record)
        print(" ".join(f"{k}={v:.4g}" for k, v in record.items() if v is not None), flush=True)
        ckpt = {"kind": args.model, "config": config, "state_dict": model.state_dict(),
                "epoch": epoch, "args": vars(args), **extra}
        if val["loss"] < best:
            best, best_epoch = val["loss"], epoch
            torch.save(ckpt, path)
        if epoch in args.save_epochs:
            torch.save(ckpt, os.path.join(out_dir, f"epoch_{epoch}.pt"))
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"early stopping: no improvement for {args.patience} epochs", flush=True)
            break

    model, ckpt = load_model(path, DEVICE)
    ckpt["test"] = run_epoch(model, loaders["test"], step)
    ckpt["stats"] = {"parameters": count_parameters(model), "best_epoch": ckpt["epoch"],
                     "train_time_s": time.time() - start,
                     "peak_train_vram_mb": max(h["peak_vram_mb"] or 0 for h in history)}
    torch.save(ckpt, path)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, os.path.join(out_dir, "curves.png"))
    print("test:", ckpt["test"])
    print("stats:", ckpt["stats"])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.model == "prior":
        vqvae, vq_ckpt = load_model(args.vqvae_ckpt, DEVICE)
        vq_args = vq_ckpt["args"]
        datasets = get_datasets(vq_args["data_root"], vq_args["max_slices"], seed=vq_args["seed"])
        codes = {s: torch.cat([vqvae.encode_indices(x.to(DEVICE)).cpu() for x in DataLoader(d, 64)])
                 for s, d in datasets.items()}
        if args.augment:
            mirrored = [vqvae.encode_indices(x.flip(-2).to(DEVICE)).cpu() for x in DataLoader(datasets["train"], 64)]
            codes["train"] = torch.cat([codes["train"], *mirrored])
        loaders = {s: DataLoader(c, args.batch_size, shuffle=s == "train", drop_last=s == "train")
                   for s, c in codes.items()}
        config = dict(num_embeddings=vqvae.quantizer.num_embeddings, dim=args.prior_dim,
                      n_layers=args.prior_layers, dropout=args.prior_dropout)
        extra = {"grid_shape": tuple(codes["train"].shape[1:]), "vqvae_ckpt": os.path.abspath(args.vqvae_ckpt)}

        def step(model, batch, epoch):
            return prior_step(model, batch)
    else:
        loaders, _ = get_dataloaders(args.batch_size, args.num_workers, data_root=args.data_root,
                                     max_slices=args.max_slices, augment=args.augment, seed=args.seed)
        if args.model == "vqvae":
            config = dict(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim,
                          downsample=args.downsample)
        else:
            config = dict(latent_dim=args.embedding_dim, downsample=args.downsample, image_shape=IMAGE_SHAPE)
        extra = {}

        def step(model, batch, epoch):
            warmup = 1.0 if epoch is None else min(1.0, epoch / args.kl_warmup)
            return autoencoder_step(model, args.model, batch, args.kl_weight * warmup)

    model = MODELS[args.model](**config).to(DEVICE)
    print(f"{args.model}: {count_parameters(model):,} parameters on {DEVICE}")
    fit(model, config, loaders, step, args, extra)


if __name__ == "__main__":
    main()
