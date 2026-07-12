#!/usr/bin/env python3
"""Export the generated image samples that the reported FID values are computed from.

The editor of The Visual Computer asked for the generated samples used for the
metrics. Because those sample sets are large, we do not ship them as static
binaries; instead this script regenerates them *bit-for-bit* from the fixed
random seeds and the released checkpoints/schedules, and writes them to disk as
PNG grids plus a raw .npz tensor. Running it reproduces exactly the image sets
that were fed to the Inception network for Tables 1-3.

Outputs (under --outdir):
    mnist_T{T}_seed{S}_ddpm.npz / .png     unconditional MNIST samples
    cifar_T{T}_seed{S}_ddpm.npz / .png     unconditional CIFAR-10 samples
    cifar_anchor_{sampler}_{steps}.npz/.png  DDPM vs DDIM at reduced step budgets
    manifest.json                          seed, T, sampler, step count, SHA-256 of each file

Usage:
    python scripts/export_samples.py --dataset mnist --T 700 --seed 0 --n 3000
    python scripts/export_samples.py --dataset cifar --T 500 --seed 0 --n 2000
    python scripts/export_samples.py --dataset cifar --anchor --sampler ddim --steps 50
"""
import argparse, hashlib, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _save_grid(imgs, path, nrow=16):
    """imgs: torch tensor in [-1, 1], shape (N, C, H, W). Saves a PNG contact grid."""
    from torchvision.utils import save_image
    save_image((imgs[:nrow * nrow] + 1) / 2, path, nrow=nrow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mnist', 'cifar'], required=True)
    ap.add_argument('--T', type=int, default=1000, help='training-time diffusion length')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=None, help='number of samples (default: paper setting)')
    ap.add_argument('--epochs', type=int, default=None, help='training budget (default: paper setting)')
    ap.add_argument('--anchor', action='store_true', help='CIFAR sampler experiment (T=1000 anchor)')
    ap.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm')
    ap.add_argument('--steps', type=int, default=None, help='restoration steps (default: T)')
    ap.add_argument('--outdir', default='results/samples')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.dataset == 'mnist':
        from src.mnist_diffusion import (set_seed, mnist_loader, SimpleUNet, make_schedule,
                                         sample_ddpm, sample_ddim, device)
        n = args.n or 3000
        epochs = args.epochs or 60
        set_seed(args.seed)
        _, loader = mnist_loader(batch_size=128, train=True)
        set_seed(args.seed)
        schedule = make_schedule(args.T)
        model = SimpleUNet().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)
        for ep in range(epochs):
            model.train()
            for x, _ in loader:
                x = x.to(device)
                t = torch.randint(0, args.T, (x.size(0),), device=device)
                noise = torch.randn_like(x)
                x_t = (schedule['sqrt_alpha_bar'][t].view(-1, 1, 1, 1) * x +
                       schedule['sqrt_one_minus_alpha_bar'][t].view(-1, 1, 1, 1) * noise)
                loss = torch.mean((noise - model(x_t, t)) ** 2)
                opt.zero_grad(); loss.backward(); opt.step()
            print(f'  epoch {ep + 1}/{epochs}', flush=True)
    else:
        from src.cifar_diffusion import (set_seed, cifar_loader, train_ddpm, make_steps,
                                         sample_ddpm, sample_ddim, device)
        n = args.n or 2000
        epochs = args.epochs or 30
        T = 1000 if args.anchor else args.T
        set_seed(args.seed)
        _, loader = cifar_loader(batch_size=64, train=True)
        set_seed(args.seed)
        model, schedule = train_ddpm(T, epochs, loader)
        args.T = T

    # ---- sample with the requested sampler / step budget ----
    if args.dataset == 'mnist':
        from src.mnist_diffusion import make_steps
    steps_n = args.steps or args.T
    steps = make_steps(args.T, steps_n)
    batches, done = [], 0
    while done < n:
        cur = min(128, n - done)
        if args.sampler == 'ddim':
            imgs = sample_ddim(model, schedule, cur, steps)
        else:
            imgs = sample_ddpm(model, schedule, cur, steps=steps if steps_n != args.T else None)
        batches.append(imgs.cpu())
        done += cur
        print(f'  sampled {done}/{n}', flush=True)
    imgs = torch.cat(batches)[:n]

    tag = (f'{args.dataset}_anchor_{args.sampler}_{steps_n}' if args.anchor
           else f'{args.dataset}_T{args.T}_seed{args.seed}_{args.sampler}')
    npz = os.path.join(args.outdir, tag + '.npz')
    png = os.path.join(args.outdir, tag + '.png')
    np.savez_compressed(npz, samples=imgs.numpy().astype(np.float16))
    _save_grid(imgs, png)

    mpath = os.path.join(args.outdir, 'manifest.json')
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest[tag] = dict(dataset=args.dataset, T=args.T, seed=args.seed, epochs=epochs,
                         sampler=args.sampler, restoration_steps=steps_n, n_samples=n,
                         npz_sha256=_sha256(npz), png_sha256=_sha256(png))
    json.dump(manifest, open(mpath, 'w'), indent=2)
    print('\nWrote', npz, '\nWrote', png, '\nUpdated', mpath)


if __name__ == '__main__':
    main()
