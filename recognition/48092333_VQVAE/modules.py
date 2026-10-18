import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, res_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, res_channels, 3, padding=1),
            nn.ReLU(), nn.Conv2d(res_channels, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Sequential):
    """Strided 4x4 convolutions then residual blocks; downsamples H and W by `downsample`."""

    def __init__(self, in_channels=1, hidden=128, res_hidden=64, n_res=2, downsample=4):
        stages = int(math.log2(downsample))
        layers, ch = [], in_channels
        for i in range(stages):
            out = hidden if i == stages - 1 else hidden // 2
            layers += [nn.Conv2d(ch, out, 4, stride=2, padding=1), nn.ReLU()]
            ch = out
        layers.append(nn.Conv2d(hidden, hidden, 3, padding=1))
        layers += [ResidualBlock(hidden, res_hidden) for _ in range(n_res)]
        super().__init__(*layers, nn.ReLU())


class Decoder(nn.Sequential):
    """Residual blocks then transposed 4x4 convolutions; sigmoid output in [0, 1]."""

    def __init__(self, in_channels, hidden=128, res_hidden=64, n_res=2, downsample=4, out_channels=1):
        stages = int(math.log2(downsample))
        layers = [nn.Conv2d(in_channels, hidden, 3, padding=1)]
        layers += [ResidualBlock(hidden, res_hidden) for _ in range(n_res)]
        layers.append(nn.ReLU())
        ch = hidden
        for i in range(stages):
            last = i == stages - 1
            out = out_channels if last else hidden // 2
            layers.append(nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1))
            layers.append(nn.Sigmoid() if last else nn.ReLU())
            ch = out
        super().__init__(*layers)

class VectorQuantizerEMA(nn.Module):
    """Nearest-neighbour codebook updated by exponential moving averages.

    Codes are initialised from the first batch of encoder outputs, and every
    `restart_every` steps any code that was not used in that window is moved to a
    random current encoder output, which prevents codebook collapse.
    """

    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25,
                 decay=0.99, eps=1e-5, restart_every=20):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.restart_every = restart_every
        self.register_buffer("embedding", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_sum", torch.zeros(num_embeddings, embedding_dim))
        self.register_buffer("usage", torch.zeros(num_embeddings))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def forward(self, z):
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        if self.training and self.steps == 0:
            self._restart(torch.ones_like(self.usage, dtype=torch.bool), flat.detach())
        distances = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embedding.t()
                     + self.embedding.pow(2).sum(1))
        indices = distances.argmin(1)
        quantized = self.embedding[indices]
        if self.training:
            self._update(flat.detach(), indices)

        commitment = self.commitment_cost * F.mse_loss(flat, quantized.detach())
        quantized = flat + (quantized - flat).detach()
        probs = torch.bincount(indices, minlength=self.num_embeddings).float() / indices.numel()
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        z_q = quantized.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        return z_q, commitment, perplexity, indices.view(b, h, w)

    @torch.no_grad()
    def _update(self, flat, indices):
        counts = torch.bincount(indices, minlength=self.num_embeddings).float()
        sums = torch.zeros_like(self.ema_sum).index_add_(0, indices, flat)
        self.ema_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        n = self.ema_size.sum()
        size = (self.ema_size + self.eps) / (n + self.num_embeddings * self.eps) * n
        self.embedding.copy_(self.ema_sum / size[:, None])

        self.usage += counts
        self.steps += 1
        if self.steps % self.restart_every == 0:
            self._restart(self.usage == 0, flat)
            self.usage.zero_()

    @torch.no_grad()
    def _restart(self, dead, flat):
        n = int(dead.sum())
        if n:
            new = flat[torch.randint(0, flat.shape[0], (n,), device=flat.device)]
            self.embedding[dead] = new
            self.ema_sum[dead] = new
            self.ema_size[dead] = 1.0

    def lookup(self, indices):
        return self.embedding[indices].permute(0, 3, 1, 2).contiguous()

class VQVAE(nn.Module):
    def __init__(self, hidden=128, res_hidden=64, n_res=2, num_embeddings=512,
                 embedding_dim=64, downsample=4, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.encoder = Encoder(1, hidden, res_hidden, n_res, downsample)
        self.pre_quant = nn.Conv2d(hidden, embedding_dim, 1)
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        self.decoder = Decoder(embedding_dim, hidden, res_hidden, n_res, downsample)

    def encode(self, x):
        return self.pre_quant(self.encoder(x))

    def forward(self, x):
        z_q, commitment, perplexity, indices = self.quantizer(self.encode(x))
        return self.decoder(z_q), commitment, perplexity, indices

    @torch.no_grad()
    def encode_indices(self, x):
        return self.quantizer(self.encode(x))[3]

    def decode_indices(self, indices):
        return self.decoder(self.quantizer.lookup(indices))

    def decode_latents(self, z_e):
        return self.decoder(self.quantizer(z_e)[0])

class VAE(nn.Module):
    """Baseline: same trunk as the VQ-VAE with a diagonal Gaussian latent."""

    def __init__(self, hidden=128, res_hidden=64, n_res=2, latent_dim=64, downsample=4,
                 image_shape=(256, 128)):
        super().__init__()
        self.latent_shape = (latent_dim, image_shape[0] // downsample, image_shape[1] // downsample)
        self.encoder = Encoder(1, hidden, res_hidden, n_res, downsample)
        self.to_moments = nn.Conv2d(hidden, 2 * latent_dim, 1)
        self.decoder = Decoder(latent_dim, hidden, res_hidden, n_res, downsample)

    def encode(self, x):
        mu, logvar = self.to_moments(self.encoder(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        return self.decoder(torch.randn((n,) + self.latent_shape, device=device))

MODELS = {"vae": VAE, "vqvae": VQVAE}

def load_model(path, device="cpu"):
    """Rebuild a model from a checkpoint written by train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MODELS[ckpt["kind"]](**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt