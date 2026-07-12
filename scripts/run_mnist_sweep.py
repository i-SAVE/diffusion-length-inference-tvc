#!/usr/bin/env python3
"""Reproduce Table 1 / Fig. 3: MNIST FID vs diffusion length T at 10/30/60 epochs.

Trains one model per (T, seed) once to the largest budget and evaluates FID at
each training-budget checkpoint. Writes results/mnist/budget_sweep.json.

Usage:
    python scripts/run_mnist_sweep.py                 # full sweep (paper setting)
    python scripts/run_mnist_sweep.py --smoke         # 1 seed, 1 epoch (fast check)
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.mnist_diffusion import (set_seed, mnist_loader, SimpleUNet, make_schedule,
                                 sample_ddpm, fid_for_generator, get_inception, device)

CHECKPOINTS = [10, 30, 60]
STEP_COUNTS = [100, 200, 400, 700, 1000]
SEEDS = [0, 1, 2]
N_REAL = N_GEN = 3000


def train_and_eval_checkpoints(T, checkpoints, loader, train_ds, inception, lr=2e-4):
    schedule = make_schedule(T)
    model = SimpleUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cps = sorted(checkpoints)
    out = {}
    for ep in range(1, cps[-1] + 1):
        model.train()
        for x, _ in loader:
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            x_t = (schedule['sqrt_alpha_bar'][t].view(-1, 1, 1, 1) * x +
                   schedule['sqrt_one_minus_alpha_bar'][t].view(-1, 1, 1, 1) * noise)
            loss = torch.mean((noise - model(x_t, t)) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in cps:
            fid = fid_for_generator(lambda bs: sample_ddpm(model, schedule, bs),
                                    train_ds, N_REAL, N_GEN, inception)
            out[ep] = fid
            print(f'    epoch {ep:>3}: FID={fid:.1f}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='1 seed / 1 epoch smoke test')
    ap.add_argument('--out', default='results/mnist/budget_sweep.json')
    args = ap.parse_args()

    checkpoints, seeds = (CHECKPOINTS, SEEDS)
    if args.smoke:
        checkpoints, seeds = [1], [0]

    inception = get_inception()
    print('device:', device, '| checkpoints:', checkpoints, '| seeds:', seeds, flush=True)

    runs = {ep: {T: [] for T in STEP_COUNTS} for ep in checkpoints}
    for seed in seeds:
        set_seed(seed)
        _, loader = mnist_loader(batch_size=128, train=True)
        for T in STEP_COUNTS:
            set_seed(seed)  # identical init/data order for every T -> T is the only variable
            print(f'seed={seed} T={T}', flush=True)
            res = train_and_eval_checkpoints(T, checkpoints, loader, loader.dataset, inception)
            for ep in checkpoints:
                runs[ep][T].append(res[ep])

    summary = {}
    for ep in checkpoints:
        print(f'\n=== Budget {ep} epochs: FID mean +/- s.d. over seeds ===')
        summary[str(ep)] = {}
        for T in STEP_COUNTS:
            a = np.array(runs[ep][T])
            m = float(a.mean())
            s = float(a.std(ddof=1)) if len(a) > 1 else 0.0
            summary[str(ep)][str(T)] = (m, s)
            print(f'  T={T:>4}: {m:6.1f} +/- {s:4.1f}   runs={np.round(a, 1).tolist()}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, 'w'), indent=2)
    print('\nSaved', args.out)


if __name__ == '__main__':
    main()
