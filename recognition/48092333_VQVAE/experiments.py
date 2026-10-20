"""Side experiments behind the README's ablations; the main pipeline does not need them.

    python experiments.py regularisation --vqvae-ckpt outputs/vqvae/best.pt
    python experiments.py regularisation --variants mirror+dropout+wd+dim64
    python experiments.py vae-sweep

regularisation: trains five PixelCNN priors, adding one regulariser at a time, and
    compares them on validation bits per code (the selection metric). The unregularised
    and final priors are also audited for memorisation at several training epochs.
vae-sweep: trains VAE baselines at KL weights 1, 0.1 and 0.01 and benchmarks each on the
    test patients with the same reference set.

Each writes a markdown table plus every run's checkpoint, history and figures to
<out-dir>/<experiment>/.
"""

import argparse
import json
import os

import torch

import train
from dataset import get_datasets
from modules import MODELS, load_model
from predict import (InceptionFeatures, audit_prior, batched, density_coverage, frechet_distance, generation_stats,
                     image_grid, nearest_ssim, summarise)
from train import DEVICE, psnr, ssim

PRIOR_VARIANTS = {
    "none": dict(mirror=False, dropout=0.0, weight_decay=0.0, dim=128),
    "mirror": dict(mirror=True, dropout=0.0, weight_decay=0.0, dim=128),
    "mirror+dropout": dict(mirror=True, dropout=0.2, weight_decay=0.0, dim=128),
    "mirror+dropout+wd": dict(mirror=True, dropout=0.2, weight_decay=0.01, dim=128),
    "mirror+dropout+wd+dim64": dict(mirror=True, dropout=0.2, weight_decay=0.01, dim=64),
}
AUDITED_VARIANTS = ("none", "mirror+dropout+wd+dim64")
AUDIT_EPOCHS = (8, 20, 25, 50, 100)
KL_WEIGHTS = (1.0, 0.1, 0.01)


def write_table(out, name, lines):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def regularisation(out, vqvae_ckpt, train_args, variants=None, **audit_args):
    lines = ["| Prior | Best epoch | Min validation bits/code | Test bits/code | Final val - train gap "
             "| Parameters | s/epoch |", "|---|---|---|---|---|---|---|"]
    memorisation = ["| Prior | Epoch | Validation bits/code | Possible copies | Near-duplicates | Mean NN-SSIM "
                    "| Max NN-SSIM | Pairwise SSIM |", "|---|---|---|---|---|---|---|---|"]
    for name, v in PRIOR_VARIANTS.items():
        if variants and name not in variants:
            continue
        run = os.path.join(out, name)
        save = ["--save-epochs", *map(str, AUDIT_EPOCHS)] if name in AUDITED_VARIANTS else []
        args = train.parse_args(["--model", "prior", "--vqvae-ckpt", vqvae_ckpt, "--out-dir", run,
                                 "--prior-dim", str(v["dim"]), *save, *train_args])
        torch.manual_seed(args.seed)
        loaders, extra, num_embeddings = train.prior_loaders(vqvae_ckpt, args.batch_size, v["mirror"])
        config = dict(num_embeddings=num_embeddings, dim=v["dim"], n_layers=args.prior_layers, dropout=v["dropout"])
        model = MODELS["prior"](**config).to(DEVICE)
        train.fit(model, config, loaders, lambda m, b, e: train.prior_step(m, b), args, extra, v["weight_decay"])
        with open(os.path.join(run, "prior", "history.json")) as f:
            history = json.load(f)
        ckpt = torch.load(os.path.join(run, "prior", "best.pt"), map_location="cpu", weights_only=False)
        best = min(history, key=lambda r: r["val_bits_per_code"])
        gap = history[-1]["val_bits_per_code"] - history[-1]["train_bits_per_code"]
        lines.append(f"| {name} | {best['epoch']} | {best['val_bits_per_code']:.3f} | "
                     f"{ckpt['test']['bits_per_code']:.3f} | {gap:.2f} | {ckpt['stats']['parameters']:,} | "
                     f"{sum(r['epoch_s'] for r in history) / len(history):.0f} |")
        for epoch in AUDIT_EPOCHS if name in AUDITED_VARIANTS else ():
            r = audit_prior(vqvae_ckpt, os.path.join(run, "prior", f"epoch_{epoch}.pt"), **audit_args)
            with open(os.path.join(run, "prior", f"memorisation_epoch_{epoch}.json"), "w") as f:
                json.dump(r, f, indent=2)
            memorisation.append(f"| {name} | {epoch} | {history[epoch - 1]['val_bits_per_code']:.3f} | "
                                f"{100 * r['frac_possible_copy']:.1f}% | {100 * r['frac_near_duplicate']:.1f}% | "
                                f"{r['nn_ssim_train']['mean']:.3f} | {r['nn_ssim_train']['max']:.3f} | "
                                f"{r['pairwise_ssim']:.3f} |")
    write_table(out, "regularisation.md", lines)
    write_table(out, "memorisation.md", memorisation)


@torch.no_grad()
def evaluate_vae(path, out, num_samples=1000, batch_size=100, seed=0, fid=True):
    """Benchmark one VAE checkpoint on the test patients (reconstruction, samples, memorisation, FID)."""
    torch.manual_seed(seed)
    vae, ckpt = load_model(path, DEVICE)
    a, name = ckpt["args"], f"vae_kl{ckpt['args']['kl_weight']:g}"
    data = get_datasets(a["data_root"], a["max_slices"], splits=("train", "test"), seed=a["seed"])
    train_images, test = data["train"].images, data["test"].images
    rec = batched(lambda x: vae(x.float())[0], test)
    scores = ssim(rec, test)
    sizes = [min(batch_size, num_samples - i) for i in range(0, num_samples, batch_size)]
    samples = torch.cat([vae.sample(k, DEVICE).cpu() for k in sizes])
    threshold = torch.quantile(nearest_ssim(test, train_images)[0], 0.99).item()
    result = {"checkpoint": path, "kl_weight": a["kl_weight"], "epoch": ckpt["epoch"],
              "reconstruction": {"ssim": summarise(scores), "psnr": summarise(psnr(rec, test)),
                                 "frac_ssim_above_0.6": (scores > 0.6).float().mean().item()},
              "generation": generation_stats(nearest_ssim(samples, train_images)[0], samples, threshold)}
    if fid:
        inception = InceptionFeatures().to(DEVICE)
        reference = inception(train_images[torch.randperm(len(train_images))[:2000]])
        features = inception(samples)
        result["generation"].update(fid=frechet_distance(features, reference),
                                    **density_coverage(reference, features))
    idx = torch.linspace(0, len(test) - 1, 8).long()
    image_grid([[test[i] for i in idx], [rec[i] for i in idx], list(samples[:8]), list(samples[8:16])],
               [[f"test {i}" for i in idx], [f"recon SSIM {scores[i]:.3f}" for i in idx]] + [[""] * 8] * 2,
               os.path.join(out, f"{name}.png"),
               f"VAE (KL weight {a['kl_weight']:g}): test slices, reconstructions, 16 uncurated samples")
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def vae_sweep(out, train_args, fid=True, **evaluate_args):
    lines = ["| KL weight | Reconstruction SSIM | Slices with SSIM > 0.6 | Sample NN-SSIM | Pairwise SSIM "
             "| FID | Density | Coverage |", "|---|---|---|---|---|---|---|---|"]
    for w in KL_WEIGHTS:
        train.main(["--model", "vae", "--kl-weight", f"{w:g}", "--out-dir", out, *train_args])
        r = evaluate_vae(os.path.join(out, f"vae_kl{w:g}", "best.pt"), out, fid=fid, **evaluate_args)
        rec, g = r["reconstruction"], r["generation"]
        lines.append(f"| {w:g} | {rec['ssim']['mean']:.3f} | {100 * rec['frac_ssim_above_0.6']:.1f}% | "
                     f"{g['nn_ssim_train']['mean']:.3f} | {g['pairwise_ssim']:.3f} | "
                     + (f"{g['fid']:.1f} | {g['density']:.3f} | {g['coverage']:.3f} |" if fid else "n/a | n/a | n/a |"))
    write_table(out, "vae_sweep.md", lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=["regularisation", "vae-sweep"])
    p.add_argument("--out-dir", default="outputs_experiments")
    p.add_argument("--vqvae-ckpt", default="outputs/vqvae/best.pt")
    p.add_argument("--variants", nargs="*", choices=list(PRIOR_VARIANTS), help="only these priors (default: all)")
    args = p.parse_args()
    torch.backends.cudnn.benchmark = True
    if args.experiment == "regularisation":
        regularisation(os.path.join(args.out_dir, "regularisation"), args.vqvae_ckpt, ["--epochs", "100", "--lr", "3e-4"],
                       args.variants)
    else:
        vae_sweep(os.path.join(args.out_dir, "vae_sweep"), ["--epochs", "50"])
