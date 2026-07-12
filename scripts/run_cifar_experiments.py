#!/usr/bin/env python3
"""Reproduce Tables 2-3 / Figs. 4 and 7 on CIFAR-10.

Experiment A (Table 2 / Fig. 4): FID vs diffusion length T, mean +/- s.d. over seeds.
Experiment B (Table 3 / Fig. 7): DDPM vs DDIM FID across restoration-step counts,
                                 for the T = 1000 anchor model.

Usage:
    python scripts/run_cifar_experiments.py                    # paper setting (LIGHT preset)
    python scripts/run_cifar_experiments.py --preset FULL      # larger sweep
    python scripts/run_cifar_experiments.py --smoke            # fast check
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.cifar_diffusion import (set_seed, cifar_loader, train_ddpm, make_steps,
                                 sample_ddpm, sample_ddim, fid_for_generator,
                                 get_inception, device)

PRESETS = {
    # the paper uses LIGHT
    'LIGHT': dict(EPOCHS=30, SEEDS=[0, 1], STEP_COUNTS=[200, 500, 1000], N=2000, BATCH=64),
    'FULL':  dict(EPOCHS=60, SEEDS=[0, 1, 2], STEP_COUNTS=[100, 200, 400, 700, 1000], N=5000, BATCH=128),
    'SMOKE': dict(EPOCHS=1, SEEDS=[0], STEP_COUNTS=[200, 1000], N=256, BATCH=64),
}
SAMPLER_STEPS = [1000, 500, 200, 100, 50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='LIGHT', choices=list(PRESETS))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--outdir', default='results/cifar')
    args = ap.parse_args()
    cfg = PRESETS['SMOKE' if args.smoke else args.preset]

    EPOCHS, SEEDS = cfg['EPOCHS'], cfg['SEEDS']
    STEP_COUNTS, BATCH = cfg['STEP_COUNTS'], cfg['BATCH']
    N_REAL = N_GEN = cfg['N']
    os.makedirs(args.outdir, exist_ok=True)

    inception = get_inception()
    print(f'preset: {args.preset} | device: {device} | epochs={EPOCHS} seeds={SEEDS}', flush=True)

    # ---- anchor model T=1000, seed SEEDS[0] (shared by both experiments) ----
    set_seed(SEEDS[0])
    train_ds, train_loader = cifar_loader(batch_size=BATCH, train=True)
    t0 = time.time()
    anchor_model, anchor_sched = train_ddpm(1000, EPOCHS, train_loader)
    print(f'anchor T=1000 trained in {(time.time() - t0) / 60:.1f} min', flush=True)

    # ---- Experiment B: sampler x steps (Table 3) ----
    table3 = {'DDPM': {}, 'DDIM': {}}
    for s in SAMPLER_STEPS:
        steps = make_steps(1000, s)
        fd = fid_for_generator(lambda bs: sample_ddpm(anchor_model, anchor_sched, bs, steps=steps),
                               train_ds, N_REAL, N_GEN, inception)
        fi = fid_for_generator(lambda bs: sample_ddim(anchor_model, anchor_sched, bs, steps),
                               train_ds, N_REAL, N_GEN, inception)
        table3['DDPM'][str(s)] = fd
        table3['DDIM'][str(s)] = fi
        print(f'steps={s:>4}:  DDPM FID={fd:6.1f}   DDIM FID={fi:6.1f}', flush=True)
        json.dump(table3, open(os.path.join(args.outdir, 'cifar_table2_sampler_fid.json'), 'w'), indent=2)

    # ---- Experiment A: FID vs T (Table 2) ----
    fid_runs = {T: [] for T in STEP_COUNTS}
    trained = {(1000, SEEDS[0]): (anchor_model, anchor_sched)}
    for seed in SEEDS:
        set_seed(seed)
        _, loader = cifar_loader(batch_size=BATCH, train=True)
        for T in STEP_COUNTS:
            if (T, seed) in trained:
                model, sched = trained[(T, seed)]
            else:
                set_seed(seed)
                print(f'training T={T} seed={seed} ...', flush=True)
                model, sched = train_ddpm(T, EPOCHS, loader)
            fid = fid_for_generator(lambda bs: sample_ddpm(model, sched, bs),
                                    train_ds, N_REAL, N_GEN, inception)
            fid_runs[T].append(fid)
            print(f'seed={seed} T={T}: FID={fid:.1f}', flush=True)

    summary = {}
    print('\n=== CIFAR-10: FID mean +/- s.d. over seeds ===')
    for T in STEP_COUNTS:
        a = np.array(fid_runs[T])
        m = float(a.mean())
        sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        summary[str(T)] = (m, sd)
        print(f'T={T:>4}: {m:6.1f} +/- {sd:4.1f}   runs={np.round(a, 1).tolist()}')
    json.dump(summary, open(os.path.join(args.outdir, 'cifar_table1_fid_variance.json'), 'w'), indent=2)
    print('\nSaved JSON metric files to', args.outdir)


if __name__ == '__main__':
    main()
