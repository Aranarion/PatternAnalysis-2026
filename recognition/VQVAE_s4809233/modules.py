"""VQ-VAE, Gaussian VAE baselines and Gated PixelCNN prior (pure PyTorch).

References:
    VQ-VAE and EMA codebook: van den Oord et al. 2017
    Gated PixelCNN: van den Oord et al. 2016
    VAE: Kingma & Welling 2014
    Codebook restarts: Dhariwal et al. 2020 (Jukebox)
    Dropout in PixelCNNs: Salimans et al. 2017 (PixelCNN++)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """x + Conv1x1(ReLU(Conv3x3(ReLU(x)))), the residual block of the VQ-VAE paper."""

    def __init__(self, channels, res_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, res_channels, 3, padding=1),
            nn.ReLU(), nn.Conv2d(res_channels, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Sequential):
    """Strided 4x4 convolutions then residual blocks, downsamples H and W."""

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
    """Residual blocks then transposed 4x4 convolutions, sigmoid output in [0, 1]."""

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

    The codebook is initialised from the first batch of encoder outputs, and every
    `restart_every` steps any code unused in that window is moved to a random current
    encoder output, preventing codebook collapse.
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
        """z (B, D, H, W) -> (z_q, commitment loss, perplexity, code indices (B, H, W)).

        Gradients reach z through the straight-through estimator.
        """
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
        """EMA update of code counts and sums (Laplace-smoothed), then periodic restart of unused codes."""
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
        """Move the codes flagged in `dead` to randomly chosen current encoder outputs."""
        n = int(dead.sum())
        if n:
            new = flat[torch.randint(0, flat.shape[0], (n,), device=flat.device)]
            self.embedding[dead] = new
            self.ema_sum[dead] = new
            self.ema_size[dead] = 1.0

    def lookup(self, indices):
        """Code indices (B, H, W) -> codebook vectors (B, D, H, W)."""
        return self.embedding[indices].permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    """Encoder -> 1x1 projection -> vector quantiser -> decoder."""

    def __init__(self, hidden=128, res_hidden=64, n_res=2, num_embeddings=512,
                 embedding_dim=64, downsample=4, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.encoder = Encoder(1, hidden, res_hidden, n_res, downsample)
        self.pre_quant = nn.Conv2d(hidden, embedding_dim, 1)
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        self.decoder = Decoder(embedding_dim, hidden, res_hidden, n_res, downsample)

    def encode(self, x):
        """Continuous encoder outputs z_e, before quantisation."""
        return self.pre_quant(self.encoder(x))

    def forward(self, x):
        """Returns (reconstruction, commitment loss, perplexity, code indices)."""
        z_q, commitment, perplexity, indices = self.quantizer(self.encode(x))
        return self.decoder(z_q), commitment, perplexity, indices

    @torch.no_grad()
    def encode_indices(self, x):
        """Code grid (B, H, W) for a batch of slices."""
        return self.quantizer(self.encode(x))[3]

    def decode_indices(self, indices):
        """Slices from a code grid, e.g. one sampled by the PixelCNN."""
        return self.decoder(self.quantizer.lookup(indices))

    def decode_latents(self, z_e):
        """Quantise and decode arbitrary continuous latents (used for interpolation)."""
        return self.decoder(self.quantizer(z_e)[0])


class VAE(nn.Module):
    """Baseline: the same encoder/decoder with a diagonal Gaussian latent instead of a codebook."""

    def __init__(self, hidden=128, res_hidden=64, n_res=2, latent_dim=64, downsample=4,
                 image_shape=(256, 128)):
        super().__init__()
        self.latent_shape = (latent_dim, image_shape[0] // downsample, image_shape[1] // downsample)
        self.encoder = Encoder(1, hidden, res_hidden, n_res, downsample)
        self.to_moments = nn.Conv2d(hidden, 2 * latent_dim, 1)
        self.decoder = Decoder(latent_dim, hidden, res_hidden, n_res, downsample)

    def encode(self, x):
        """Mean and log-variance of q(z|x); log-variance clamped for numerical stability."""
        mu, logvar = self.to_moments(self.encoder(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-30, 20)

    def forward(self, x):
        """Reparameterised sample while training, the mean at evaluation; returns (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        """Decode n draws from N(0, I)."""
        return self.decoder(torch.randn((n,) + self.latent_shape, device=device))


class MaskedConv2d(nn.Conv2d):
    """Conv2d whose kernel is multiplied by a fixed binary mask, so masked weights never contribute."""

    def __init__(self, *args, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding)


class GatedLayer(nn.Module):
    """Gated PixelCNN layer with vertical (rows above) and horizontal (left in the row) stacks.

    The two stacks avoid the original PixelCNN's blind spot. Mask "A" (first layer only)
    also hides the current position, so no layer sees the code it predicts. Outputs are
    cropped to keep both stacks causal.
    """

    def __init__(self, dim, kernel_size, mask_type="B", residual=True, dropout=0.0):
        super().__init__()
        k, p = kernel_size, kernel_size // 2
        v_mask = torch.ones(2 * dim, dim, p + 1, k)
        h_mask = torch.ones(2 * dim, dim, 1, p + 1)
        if mask_type == "A":
            v_mask[:, :, -1] = 0
            h_mask[:, :, :, -1] = 0
        self.vertical = MaskedConv2d(dim, 2 * dim, (p + 1, k), padding=(p, p), mask=v_mask)
        self.horizontal = MaskedConv2d(dim, 2 * dim, (1, p + 1), padding=(0, p), mask=h_mask)
        self.v_to_h = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.h_out = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    @staticmethod
    def gate(x):
        """Gated activation tanh(a) * sigmoid(b)."""
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward(self, x_v, x_h):
        """Returns the updated (vertical, horizontal) stack features."""
        height, width = x_v.shape[-2:]
        v = self.vertical(x_v)[:, :, :height]
        h = self.horizontal(x_h)[:, :, :, :width]
        out_h = self.h_out(self.dropout(self.gate(h + self.v_to_h(v))))
        return self.gate(v), out_h + x_h if self.residual else out_h


class GatedPixelCNN(nn.Module):
    """Autoregressive prior over VQ-VAE code grids.

    The defaults (width 64, dropout 0.2) are the regularised configuration selected on
    validation loss.
    """

    def __init__(self, num_embeddings=512, dim=64, n_layers=15, kernel_size=5, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, dim)
        self.layers = nn.ModuleList(
            [GatedLayer(dim, 7, "A", residual=False, dropout=dropout)]
            + [GatedLayer(dim, kernel_size, dropout=dropout) for _ in range(n_layers - 1)]
        )
        self.output = nn.Sequential(nn.ReLU(), nn.Conv2d(dim, 512, 1), nn.ReLU(),
                                    nn.Conv2d(512, num_embeddings, 1))

    def forward(self, codes):
        """Code grid (B, H, W) -> logits over the codebook at every position (B, K, H, W)."""
        x = self.embedding(codes).permute(0, 3, 1, 2)
        x_v, x_h = x, x
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h)
        return self.output(x_h)

    @torch.no_grad()
    def sample(self, n, grid_shape, device, temperature=1.0, generator=None):
        """Generate n code grids one code at a time in raster order.

        Only rows up to the current one are fed in, since later rows cannot affect it.
        """
        h, w = grid_shape
        codes = torch.zeros(n, h, w, dtype=torch.long, device=device)
        for i in range(h):
            for j in range(w):
                logits = self(codes[:, : i + 1])[:, :, i, j] / temperature
                codes[:, i, j] = torch.multinomial(logits.softmax(1), 1, generator=generator)[:, 0]
        return codes


MODELS = {"vae": VAE, "vqvae": VQVAE, "prior": GatedPixelCNN}


def load_model(path, device="cpu"):
    """Rebuild a model from a checkpoint written by train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MODELS[ckpt["kind"]](**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt
