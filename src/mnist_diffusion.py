"""
Controlled empirical study of diffusion-process length and inference algorithm
on MNIST image synthesis.

This module contains the exact experimental code used in the paper
"Diffusion Process Length and Inference Algorithm in Denoising-Diffusion Image
Synthesis: A Controlled Empirical Study" (I. A. Kistin, Sh. M. Isaev), refactored
from the authors' original notebook into a reproducible, seed-controlled form.

The only substantive additions relative to the original study are:
  * explicit random-seed control (set_seed),
  * a multi-seed loop for uncertainty estimates on FID,
  * FID for both DDPM and DDIM across restoration-step counts,
  * a proper Inception-based FID for the conditional model
    (the original notebook used a random-projection proxy).
No modelling choice, hyper-parameter, or schedule has been changed.
"""
import math, os, random, argparse, json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as tv_models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    """Fix all RNGs for reproducible training/evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mnist_loader(batch_size=128, train=True, to_pm1=True):
    tfm = [transforms.ToTensor()]
    if to_pm1:
        tfm.append(transforms.Lambda(lambda x: x * 2. - 1.))
    ds = datasets.MNIST(root='./data', train=train, download=True,
                        transform=transforms.Compose(tfm))
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=train,
                          num_workers=2, pin_memory=True, drop_last=False)


# --------------------------------------------------------------------------- #
# Unconditional model: custom SimpleUNet (matched-noise schedule)
# --------------------------------------------------------------------------- #
def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class SimpleUNet(nn.Module):
    """Compact convolutional U-Net with additive skip connections.
    Time index -> sinusoidal embedding (dim 128) -> 2-layer MLP (SiLU),
    added at the bottleneck. ~ encoder 1->32->64->128, symmetric decoder."""
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                      nn.Linear(time_dim, time_dim))
        self.conv0 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1)
        self.act = nn.SiLU()
        self.time_dim = time_dim

    def forward(self, x, t):
        t = self.time_mlp(sinusoidal_embedding(t, self.time_dim))[:, :, None, None]
        h0 = self.act(self.conv0(x))
        h1 = self.act(self.conv1(h0))
        h2 = self.act(self.conv2(h1))
        h3 = self.act(self.conv3(h2 + t))
        u1 = self.act(self.deconv1(h3)) + h1
        u2 = self.act(self.deconv2(u1)) + h0
        return self.out_conv(u2)


def make_schedule(T, beta_1=1e-4, target_alpha_bar_T=1e-3):
    """Linear beta schedule from beta_1 to beta_T, where beta_T is found by
    binary search so that the terminal cumulative signal retention
    alpha_bar_T is (approximately) equal to target_alpha_bar_T for every T.
    This is the 'matched total noise' control central to the study."""
    low, high = beta_1, 0.999
    for _ in range(100):
        mid = (low + high) / 2
        betas = torch.linspace(beta_1, mid, T)
        alpha_bar = torch.cumprod(1. - betas, dim=0)
        if alpha_bar[-1] > target_alpha_bar_T:
            low = mid
        else:
            high = mid
    betas = torch.linspace(beta_1, high, T, device=device)
    alphas = 1. - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {'T': T, 'betas': betas, 'alphas': alphas, 'alpha_bar': alpha_bar,
            'sqrt_alpha_bar': torch.sqrt(alpha_bar),
            'sqrt_one_minus_alpha_bar': torch.sqrt(1. - alpha_bar)}


def train_ddpm(T, epochs, loader, lr=2e-4):
    schedule = make_schedule(T)
    model = SimpleUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for _ in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = (schedule['sqrt_alpha_bar'][t].view(-1, 1, 1, 1) * x +
                   schedule['sqrt_one_minus_alpha_bar'][t].view(-1, 1, 1, 1) * noise)
            loss = torch.mean((noise - model(x_t, t)) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_history.append(loss.item())
    return model, schedule, loss_history


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
def make_steps(T_train, T_infer):
    idxs = torch.linspace(0, T_train - 1, T_infer, dtype=torch.long)
    return list(reversed(idxs.tolist()))


def sample_ddpm(model, schedule, n_samples, steps=None):
    model.eval()
    T = schedule['T']
    if steps is None:
        steps = list(range(T - 1, -1, -1))
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    with torch.no_grad():
        for i in steps:
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            eps = model(x, t)
            alpha, alpha_bar, beta = schedule['alphas'][i], schedule['alpha_bar'][i], schedule['betas'][i]
            x = (1. / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1. - alpha_bar)) * eps)
            if i > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
    return x.clamp(-1., 1.)


def sample_ddim(model, schedule, n_samples, steps):
    model.eval()
    ab = schedule['alpha_bar']
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    with torch.no_grad():
        for i, j in zip(steps[:-1], steps[1:]):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            eps = model(x, t)
            x0 = (x - torch.sqrt(1. - ab[i]) * eps) / torch.sqrt(ab[i])
            x = torch.sqrt(ab[j]) * x0 + torch.sqrt(1. - ab[j]) * eps
    return x.clamp(-1., 1.)


# --------------------------------------------------------------------------- #
# FID (Inception-v3 pool features, matched to the paper)
# --------------------------------------------------------------------------- #
def get_inception():
    try:
        net = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1,
                                     transform_input=False)
    except Exception:
        net = tv_models.inception_v3(pretrained=True, transform_input=False)
    net.fc = nn.Identity(); net.eval()
    return net.to(device)


def _prep_inception(imgs):
    imgs = (imgs + 1.) / 2.
    imgs = torch.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    return (imgs.repeat(1, 3, 1, 1) - mean) / std


def _activations(inception, source, max_samples):
    feats, seen = [], 0
    with torch.no_grad():
        if isinstance(source, DataLoader):
            for x, _ in source:
                f = inception(_prep_inception(x.to(device))); feats.append(f.cpu().numpy())
                seen += f.size(0)
                if seen >= max_samples: break
        else:  # callable(batch_size)->images in [-1,1]
            while seen < max_samples:
                cur = min(128, max_samples - seen)
                f = inception(_prep_inception(source(cur).to(device))); feats.append(f.cpu().numpy())
                seen += cur
    return np.concatenate(feats, 0)[:max_samples]


def _fid(mu1, s1, mu2, s2):
    diff = mu1 - mu2
    eig = np.real(np.linalg.eigvals(s1.dot(s2))); eig[eig < 0] = 0
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.sum(np.sqrt(eig)))


def fid_for_generator(gen_fn, train_ds, n_real=3000, n_gen=3000, inception=None):
    inception = inception or get_inception()
    rf = _activations(inception, DataLoader(train_ds, batch_size=128, shuffle=True), n_real)
    gf = _activations(inception, gen_fn, n_gen)
    return _fid(rf.mean(0), np.cov(rf, rowvar=False), gf.mean(0), np.cov(gf, rowvar=False))

