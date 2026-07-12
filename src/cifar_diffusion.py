"""CIFAR-10 diffusion: model, matched-noise schedule, DDPM/DDIM samplers and FID.

Extracted verbatim from the executed Kaggle notebook used for the paper
(cifar10-diffusion-strong-2026-07-08). No modelling choice has been changed.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import math, random, json, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as tv_models

if torch.cuda.is_available():
    device = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)

def to_pm1(x): return x * 2.0 - 1.0

def cifar_loader(batch_size=64, train=True):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(to_pm1)])
    ds = datasets.CIFAR10(root='./data', train=train, download=True, transform=tfm)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0, drop_last=train)

def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class CIFARUNet(nn.Module):
    """Small pure-PyTorch U-Net for 32x32 RGB diffusion.
    This avoids the brittle diffusers import path on Kaggle while keeping
    the same unconditional DDPM training interface."""
    def __init__(self, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, 256)
        )
        self.conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)
        self.act = nn.SiLU()

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
    low, high = beta_1, 0.999
    for _ in range(100):
        mid = (low + high) / 2
        ab = torch.cumprod(1. - torch.linspace(beta_1, mid, T), dim=0)
        low, high = (mid, high) if ab[-1] > target_alpha_bar_T else (low, mid)
    betas = torch.linspace(beta_1, high, T, device=device)
    alphas = 1. - betas; alpha_bar = torch.cumprod(alphas, dim=0)
    return {'T': T, 'betas': betas, 'alphas': alphas, 'alpha_bar': alpha_bar,
            'sqrt_alpha_bar': torch.sqrt(alpha_bar), 'sqrt_one_minus_alpha_bar': torch.sqrt(1. - alpha_bar)}

def train_ddpm(T, epochs, loader, lr=2e-4, log=print):
    schedule = make_schedule(T); model = CIFARUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = (device.type == 'cuda'); scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for ep in range(1, epochs + 1):
        model.train(); last = 0.0
        for x, _ in loader:
            x = x.to(device); t = torch.randint(0, T, (x.size(0),), device=device); noise = torch.randn_like(x)
            x_t = (schedule['sqrt_alpha_bar'][t].view(-1,1,1,1) * x +
                   schedule['sqrt_one_minus_alpha_bar'][t].view(-1,1,1,1) * noise)
            opt.zero_grad()
            if use_amp:
                with torch.autocast(device_type='cuda'):
                    loss = F.mse_loss(model(x_t, t), noise)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = F.mse_loss(model(x_t, t), noise); loss.backward(); opt.step()
            last = float(loss.item())
        if ep % 5 == 0 or ep == epochs: log(f'      epoch {ep}/{epochs} loss={last:.4f}')
    return model, schedule

def make_steps(T_train, T_infer):
    idxs = torch.linspace(0, T_train - 1, T_infer, dtype=torch.long)
    return list(reversed(idxs.tolist()))

@torch.no_grad()
def sample_ddpm(model, schedule, n, steps=None):
    model.eval(); T = schedule['T']
    if steps is None: steps = list(range(T - 1, -1, -1))
    x = torch.randn(n, 3, 32, 32, device=device)
    for i in steps:
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = model(x, t); a, ab, b = schedule['alphas'][i], schedule['alpha_bar'][i], schedule['betas'][i]
        x = (1. / torch.sqrt(a)) * (x - (b / torch.sqrt(1. - ab)) * eps)
        if i > 0: x = x + torch.sqrt(b) * torch.randn_like(x)
    return x.clamp(-1., 1.)

@torch.no_grad()
def sample_ddim(model, schedule, n, steps):
    model.eval(); ab = schedule['alpha_bar']
    x = torch.randn(n, 3, 32, 32, device=device)
    for i, j in zip(steps[:-1], steps[1:]):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = model(x, t)
        x0 = (x - torch.sqrt(1. - ab[i]) * eps) / torch.sqrt(ab[i])
        x = torch.sqrt(ab[j]) * x0 + torch.sqrt(1. - ab[j]) * eps
    return x.clamp(-1., 1.)

def get_inception():
    try:
        net = tv_models.inception_v3(weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    except Exception:
        net = tv_models.inception_v3(pretrained=True, transform_input=False)
    net.fc = nn.Identity(); net.eval(); return net.to(device)

def _prep(imgs):
    imgs = (imgs + 1.) / 2.
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485,0.456,0.406], device=imgs.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=imgs.device).view(1,3,1,1)
    return (imgs - mean) / std

@torch.no_grad()
def _acts(inception, source, n_max):
    feats, seen = [], 0
    if isinstance(source, DataLoader):
        for x, _ in source:
            f = inception(_prep(x.to(device))); feats.append(f.float().cpu().numpy()); seen += f.size(0)
            if seen >= n_max: break
    else:
        while seen < n_max:
            cur = min(128, n_max - seen)
            f = inception(_prep(source(cur).to(device))); feats.append(f.float().cpu().numpy()); seen += cur
    return np.concatenate(feats, 0)[:n_max]

def _sqrtm_trace_product(sigma1, sigma2, eps=1e-6):
    """Return trace(sqrt(sqrt(s1) @ s2 @ sqrt(s1))) for PSD covariance matrices.
    This is algebraically equivalent to the scipy sqrtm term in FID but uses
    only numpy.linalg, which is more robust on Kaggle's prebuilt runtime."""
    sigma1 = ((sigma1 + sigma1.T) * 0.5) + np.eye(sigma1.shape[0]) * eps
    sigma2 = ((sigma2 + sigma2.T) * 0.5) + np.eye(sigma2.shape[0]) * eps
    w1, v1 = np.linalg.eigh(sigma1)
    w1 = np.clip(w1, a_min=0.0, a_max=None)
    sqrt_sigma1 = (v1 * np.sqrt(w1)) @ v1.T
    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    middle = (middle + middle.T) * 0.5
    wm = np.linalg.eigvalsh(middle)
    wm = np.clip(wm, a_min=0.0, a_max=None)
    return float(np.sqrt(wm).sum())

def _fid(mu1, sigma1, mu2, sigma2):
    """Standard Frechet Inception Distance without scipy.sqrtm dependency."""
    diff = mu1 - mu2
    tr_covmean = _sqrtm_trace_product(sigma1, sigma2)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)

def fid_for_generator(gen_fn, real_ds, n_real, n_gen, inception):
    rl = DataLoader(real_ds, batch_size=128, shuffle=True, num_workers=0)
    rf = _acts(inception, rl, n_real); gf = _acts(inception, gen_fn, n_gen)
    return _fid(rf.mean(0), np.cov(rf, rowvar=False), gf.mean(0), np.cov(gf, rowvar=False))

