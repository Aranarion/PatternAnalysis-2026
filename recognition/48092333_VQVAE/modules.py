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

MODELS = {"vae": VAE}


def load_model(path, device="cpu"):
    """Rebuild a model from a checkpoint written by train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MODELS[ckpt["kind"]](**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt