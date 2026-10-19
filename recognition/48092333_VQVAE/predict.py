"""Run the trained models on unseen test patients, generate new slices, and benchmark the VQ-VAE + PixelCNN against the VAE baseline.

    python predict.py --out-dir outputs

Writes metrics.json, results.md, cases.json and figures/ to <out-dir>/predict/.
"""

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


class InceptionFeatures(nn.Module):
    """2048-d torchvision Inception-v3 features (FID comparable within this project only)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
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


def write_markdown(m, path):
    def ms(s, d=3):
        return f"{s['mean']:.{d}f} ± {s['std']:.{d}f}"

    def pct(v):
        return f"{100 * v:.1f}%"

    def num(v, d=1):
        return "n/a" if v is None else f"{v:,.{d}f}"

    r, g, res = m["reconstruction"], m["generation"], m["resources"]
    lines = ["### Reconstruction of unseen test patients", "",
             "| Model | SSIM | PSNR (dB) | Slices with SSIM > 0.6 |", "|---|---|---|---|"]
    lines += [f"| {k.upper()} | {ms(v['ssim'])} | {ms(v['psnr'], 2)} | {pct(v['frac_ssim_above_0.6'])} |"
              for k, v in r.items()]
    rows = [("SSIM to nearest training slice", lambda s: ms(s["nn_ssim_train"])),
            ("SSIM to nearest test slice", lambda s: ms(s["nn_ssim_test"]) if "nn_ssim_test" in s else "n/a"),
            ("NN-SSIM > 0.6", lambda s: pct(s["frac_above_0.6"])),
            (f"Possible copies (NN-SSIM > {m['memorisation_threshold']:.3f})",
             lambda s: pct(s["frac_possible_copy"])),
            ("Near-duplicates (NN-SSIM > 0.95)", lambda s: pct(s["frac_near_duplicate"])),
            ("Mean pairwise SSIM (lower = more diverse)", lambda s: f"{s['pairwise_ssim']:.3f}")]
    if "fid" in g["vqvae"]:
        rows += [("FID vs training slices (relative only)", lambda s: f"{s['fid']:.1f}"),
                 ("Density", lambda s: f"{s['density']:.3f}"),
                 ("Coverage", lambda s: f"{s['coverage']:.3f}")]
    lines += ["", f"### Generation ({m['num_samples']} uncurated samples per model)", "",
              "| Metric | Real test slices | VAE | VQ-VAE + PixelCNN |", "|---|---|---|---|"]
    lines += [f"| {label} | {fn(g['test_real'])} | {fn(g['vae'])} | {fn(g['vqvae'])} |" for label, fn in rows]
    lines += ["", "### Resources", "",
              "| Model | Parameters | Peak train VRAM (MB) | Train time (min) | Reconstruction, batch 1 (ms) "
              "| Generation per image (ms) | Peak inference VRAM (MB) |", "|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        lines.append(f"| {k.upper()} | {v['parameters']:,} | {num(v.get('peak_train_vram_mb'), 0)} "
                     f"| {num(v['train_time_s'] / 60)} | {num(v.get('recon_latency_ms'), 2)} "
                     f"| {num(v.get('generation_ms_per_image'), 2)} | {num(v.get('peak_inference_vram_mb'), 0)} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--no-fid", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    out = os.path.join(args.out_dir, "predict")
    figs = os.path.join(out, "figures")
    os.makedirs(figs, exist_ok=True)

    models = {k: load_model(os.path.join(args.out_dir, k, "best.pt"), DEVICE) for k in ("vae", "vqvae", "prior")}
    vae, vqvae, prior = (models[k][0] for k in ("vae", "vqvae", "prior"))
    a = models["vqvae"][1]["args"]
    data = get_datasets(a["data_root"], a["max_slices"], splits=("train", "test"), seed=a["seed"])
    train, test = data["train"].images, data["test"].images
    train_names = [os.path.basename(p).split(".nii")[0] for p in data["train"].paths]
    test_names = [os.path.basename(p).split(".nii")[0] for p in data["test"].paths]
    m = {"num_samples": args.num_samples, "reconstruction": {}, "generation": {},
         "resources": {k: dict(models[k][1]["stats"]) for k in models}}

    recon = {}
    for key, model in (("vae", vae), ("vqvae", vqvae)):
        stats = {}
        with peak_vram(stats):
            rec = batched(lambda x: model(x.float())[0], test)
        scores = ssim(rec, test)
        recon[key] = (rec, scores)
        m["reconstruction"][key] = {"ssim": summarise(scores), "psnr": summarise(psnr(rec, test)),
                                    "frac_ssim_above_0.6": (scores > 0.6).float().mean().item()}
        x1 = test[:1].to(DEVICE).float()
        m["resources"][key].update(recon_latency_ms=time_ms(lambda: model(x1)),
                                   peak_inference_vram_mb=stats["peak_vram_mb"])

    (rec_vae, s_vae), (rec_vq, s_vq) = recon["vae"], recon["vqvae"]
    examples = torch.linspace(0, len(test) - 1, 6).long()
    print(f"{'test slice':32s} {'VAE SSIM':>9s} {'VQ-VAE SSIM':>12s}")
    for i in examples:
        print(f"{test_names[i]:32s} {s_vae[i]:9.3f} {s_vq[i]:12.3f}")

    sizes = [min(args.batch_size, args.num_samples - i) for i in range(0, args.num_samples, args.batch_size)]
    vae_samples, t_vae = timed(lambda: torch.cat([vae.sample(k, DEVICE).cpu() for k in sizes]))
    generator = torch.Generator(device=DEVICE).manual_seed(args.seed)

    def generate_vqvae():
        codes = torch.cat([prior.sample(k, models["prior"][1]["grid_shape"], DEVICE, args.temperature,
                                        generator).cpu() for k in sizes])
        return codes, batched(vqvae.decode_indices, codes)

    stats = {}
    with peak_vram(stats):
        (vq_codes, vq_samples), t_vq = timed(generate_vqvae)
    m["resources"]["vae"]["generation_ms_per_image"] = 1000 * t_vae / args.num_samples
    m["resources"]["prior"].update(generation_ms_per_image=1000 * t_vq / args.num_samples,
                                   peak_inference_vram_mb=stats["peak_vram_mb"])

    real_nn, _ = nearest_ssim(test, train)
    threshold = torch.quantile(real_nn, 0.99).item()
    m["memorisation_threshold"] = threshold
    m["generation"]["test_real"] = generation_stats(real_nn, test, threshold)
    audit = {}
    for key, samples in (("vae", vae_samples), ("vqvae", vq_samples)):
        nn_train, nn_idx = nearest_ssim(samples, train)
        audit[key] = (samples, nn_train, nn_idx)
        m["generation"][key] = generation_stats(nn_train, samples, threshold)
        m["generation"][key]["nn_ssim_test"] = summarise(nearest_ssim(samples, test)[0])

    if not args.no_fid:
        inception = InceptionFeatures().to(DEVICE)
        reference = inception(train[torch.randperm(len(train))[:2000]])
        for key, images in (("test_real", test), ("vae", vae_samples), ("vqvae", vq_samples)):
            features = inception(images)
            m["generation"][key].update(fid=frechet_distance(features, reference),
                                        **density_coverage(reference, features))

    k = vqvae.quantizer.num_embeddings
    train_codes = batched(lambda x: vqvae.encode_indices(x.float()), train)
    p_train = torch.bincount(train_codes.flatten(), minlength=k).double()
    p_gen = torch.bincount(vq_codes.flatten(), minlength=k).double()
    p_train, p_gen = p_train / p_train.sum(), p_gen / p_gen.sum()
    used = p_train > 0
    m["codebook"] = {"codes_used_train": int(used.sum()), "codes_used_generated": int((p_gen > 0).sum()),
                     "kl_train_to_generated": (p_train[used] * (p_train[used] / p_gen[used].clamp_min(1e-10)).log())
                     .sum().item()}

    for name, idx in (("reconstructions", examples), ("worst_reconstructions", s_vq.argsort()[:6])):
        image_grid([[test[i] for i in idx], [rec_vae[i] for i in idx], [rec_vq[i] for i in idx],
                    [(3 * (rec_vq[i] - test[i].float()).abs()).clamp(0, 1) for i in idx]],
                   [[test_names[i] for i in idx], [f"VAE {s_vae[i]:.3f}" for i in idx],
                    [f"VQ-VAE {s_vq[i]:.3f}" for i in idx], ["|VQ-VAE error| x3"] * len(idx)],
                   os.path.join(figs, f"{name}.png"), f"{name.replace('_', ' ').capitalize()}: input, VAE, VQ-VAE, error")

    for key, (samples, nn_train, nn_idx) in audit.items():
        image_grid([list(samples[r * 8:(r + 1) * 8]) for r in range(4)], [[""] * 8] * 4,
                   os.path.join(figs, f"samples_{key}.png"), f"{key.upper()}: first 32 samples (uncurated)")
        order = nn_train.argsort(descending=True)
        for name, idx in (("closest_to_train", order[:6]), ("least_plausible", order[-6:])):
            image_grid([[samples[i] for i in idx], [train[nn_idx[i]] for i in idx]],
                       [[f"NN-SSIM {nn_train[i]:.3f}" for i in idx], [train_names[nn_idx[i]] for i in idx]],
                       os.path.join(figs, f"{name}_{key}.png"),
                       f"{key.upper()} {name.replace('_', ' ')}: sample (top), nearest training slice (bottom)")

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in (("real test slices", real_nn), ("VAE samples", audit["vae"][1]),
                          ("VQ-VAE samples", audit["vqvae"][1])):
        ax.hist(values.numpy(), bins=40, range=(0, 1), alpha=0.5, density=True, label=label)
    ax.axvline(threshold, color="k", ls="--", label="memorisation threshold")
    ax.set(xlabel="SSIM to nearest training slice", ylabel="density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "nn_ssim_histogram.png"), dpi=150)
    plt.close(fig)

    j = next(i for i, n in enumerate(test_names) if patient_id(n) != patient_id(test_names[0]))
    ends = test[[0, j]].to(DEVICE).float()
    mu, ze = vae.encode(ends)[0], vqvae.encode(ends)
    steps = torch.linspace(0, 1, 8).tolist()
    image_grid([[vae.decoder((1 - t) * mu[:1] + t * mu[1:])[0].cpu() for t in steps],
                [vqvae.decode_latents((1 - t) * ze[:1] + t * ze[1:])[0].cpu() for t in steps]],
               [[f"VAE t={t:.2f}" for t in steps], [f"VQ-VAE t={t:.2f}" for t in steps]],
               os.path.join(figs, "interpolation.png"), "Latent interpolation between two test patients")

    cases = {"worst_reconstructions": [{"slice": test_names[i], "ssim_vqvae": s_vq[i].item(),
                                        "ssim_vae": s_vae[i].item()} for i in s_vq.argsort()[:20].tolist()],
             "least_plausible_vqvae": [{"sample": i, "nn_ssim_train": audit["vqvae"][1][i].item(),
                                        "nearest_train_slice": train_names[audit["vqvae"][2][i]]}
                                       for i in audit["vqvae"][1].argsort()[:20].tolist()]}
    for name, obj in (("cases.json", cases), ("metrics.json", m)):
        with open(os.path.join(out, name), "w") as f:
            json.dump(obj, f, indent=2)
    write_markdown(m, os.path.join(out, "results.md"))
    with open(os.path.join(out, "results.md"), encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
