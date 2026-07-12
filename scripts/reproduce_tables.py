#!/usr/bin/env python3
"""Print Tables 1-3 of the paper directly from the released metric files.

This is the fastest way for a reviewer to verify that the numbers in the
manuscript are exactly the numbers produced by the released runs: it needs no
GPU and no training, only the JSON files under results/.

Usage:
    python scripts/reproduce_tables.py
    python scripts/reproduce_tables.py --check      # assert values match the paper
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Values as printed in the manuscript (used by --check)
PAPER_T1 = {
    '10': {'100': 150.0, '200': 144.8, '400': 144.1, '700': 139.9, '1000': 148.5},
    '30': {'100': 117.4, '200': 110.7, '400': 101.2, '700': 93.4, '1000': 99.5},
    '60': {'100': 84.3, '200': 69.1, '400': 64.3, '700': 59.0, '1000': 63.8},
}
PAPER_T2 = {'200': 131.5, '500': 113.6, '1000': 117.0}
PAPER_T3 = {'DDPM': {'1000': 111.6, '500': 339.2, '200': 404.7, '100': 457.2, '50': 494.8},
            'DDIM': {'1000': 116.3, '500': 113.7, '200': 109.3, '100': 105.8, '50': 99.1}}


def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    t1 = load('results/mnist/budget_sweep.json')
    t2 = load('results/cifar/cifar_table1_fid_variance.json')
    t3 = load('results/cifar/cifar_table2_sampler_fid.json')

    Ts = ['100', '200', '400', '700', '1000']
    print('\nTable 1  MNIST FID (mean +/- s.d. over 3 seeds)')
    print('  epochs\\T  ' + '  '.join(f'{T:>12}' for T in Ts))
    for ep in ['10', '30', '60']:
        row = '  '.join(f'{t1[ep][T][0]:5.1f}+/-{t1[ep][T][1]:<4.1f}' for T in Ts)
        print(f'  {ep:>6}    {row}')

    print('\nTable 2  CIFAR-10 FID vs T (mean +/- s.d. over 2 seeds, 30 epochs)')
    Tc = ['200', '500', '1000']
    print('  T        ' + '  '.join(f'{T:>12}' for T in Tc))
    print('  FID      ' + '  '.join(f'{t2[T][0]:5.1f}+/-{t2[T][1]:<4.1f}   ' for T in Tc))

    print('\nTable 3  CIFAR-10 FID, DDPM vs DDIM across restoration steps (T = 1000 anchor)')
    steps = ['1000', '500', '200', '100', '50']
    print('  steps    ' + '  '.join(f'{s:>8}' for s in steps))
    for name in ('DDPM', 'DDIM'):
        print(f'  {name}     ' + '  '.join(f'{t3[name][s]:8.1f}' for s in steps))

    if args.check:
        bad = []
        for ep, row in PAPER_T1.items():
            for T, v in row.items():
                if abs(t1[ep][T][0] - v) > 0.05:
                    bad.append(f'Table1[{ep}][{T}]: file={t1[ep][T][0]:.1f} paper={v}')
        for T, v in PAPER_T2.items():
            if abs(t2[T][0] - v) > 0.05:
                bad.append(f'Table2[{T}]: file={t2[T][0]:.1f} paper={v}')
        for s_name, row in PAPER_T3.items():
            for s, v in row.items():
                if abs(t3[s_name][s] - v) > 0.05:
                    bad.append(f'Table3[{s_name}][{s}]: file={t3[s_name][s]:.1f} paper={v}')
        if bad:
            print('\nMISMATCH between released metrics and the manuscript:')
            for b in bad:
                print('  -', b)
            sys.exit(1)
        print('\nOK: every value in Tables 1-3 matches the released metric files.')


if __name__ == '__main__':
    main()
