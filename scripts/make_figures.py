#!/usr/bin/env python3
"""Regenerate Figs. 3, 4 and 7 (EPS vector + 600 dpi PNG) per Springer TVC artwork specs.

Run from the repository root:  python scripts/make_figures.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# ---- Springer TVC skill settings ----
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.1,
    'grid.linewidth': 0.4,
    'savefig.dpi': 600,
    'figure.dpi': 120,
})
COL_W = 3.307  # 84 mm single column

# Okabe-Ito colorblind-safe
BLUE, ORANGE, GREEN, VERM = '#0072B2', '#E69F00', '#009E73', '#D55E00'

# ============ Fig 3: MNIST FID vs T, three budgets ============
bs = json.load(open('results/mnist/budget_sweep.json'))
Ts = [100, 200, 400, 700, 1000]
series = [
    ('10', '10 epochs', BLUE,   'o', '-'),
    ('30', '30 epochs', ORANGE, 's', '--'),
    ('60', '60 epochs', GREEN,  '^', ':'),
]
fig, ax = plt.subplots(figsize=(COL_W, 2.45))
for key, lab, c, mk, ls in series:
    m = [bs[key][str(T)][0] for T in Ts]
    s = [bs[key][str(T)][1] for T in Ts]
    ax.errorbar(Ts, m, yerr=s, color=c, marker=mk, linestyle=ls,
                ms=3.5, capsize=2.5, elinewidth=0.8, capthick=0.8,
                markeredgewidth=0.6, label=lab, clip_on=False)
ax.set_xlabel('Number of diffusion steps $T$')
ax.set_ylabel('FID')
ax.set_xscale('log')
ax.xaxis.set_major_locator(FixedLocator(Ts))
ax.set_xticklabels([str(t) for t in Ts])
ax.minorticks_off()
ax.set_ylim(40, 170)
ax.legend(frameon=False, handlelength=2.4, loc='lower left',
          borderaxespad=0.2, labelspacing=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig('figures/Fig3.png', bbox_inches='tight')
fig.savefig('figures/Fig3.eps', bbox_inches='tight')
plt.close(fig)

# ============ Fig 4: CIFAR FID vs T ============
c1 = json.load(open('results/cifar/cifar_table1_fid_variance.json'))
Tc = [200, 500, 1000]
m = [c1[str(T)][0] for T in Tc]
s = [c1[str(T)][1] for T in Tc]
fig, ax = plt.subplots(figsize=(COL_W, 2.3))
ax.errorbar(Tc, m, yerr=s, color=VERM, marker='D', linestyle='-',
            ms=3.8, capsize=2.5, elinewidth=0.8, capthick=0.8,
            markeredgewidth=0.6, clip_on=False)
for x, y in zip(Tc, m):
    ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                xytext=(0, 6), ha='center', fontsize=8)
ax.set_xlabel('Number of diffusion steps $T$')
ax.set_ylabel('FID')
ax.set_xscale('log')
ax.xaxis.set_major_locator(FixedLocator(Tc))
ax.set_xticklabels([str(t) for t in Tc])
ax.minorticks_off()
ax.set_ylim(105, 142)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig('figures/Fig4.png', bbox_inches='tight')
fig.savefig('figures/Fig4.eps', bbox_inches='tight')
plt.close(fig)

# ============ Fig 7: CIFAR DDPM vs DDIM across steps ============
c2 = json.load(open('results/cifar/cifar_table2_sampler_fid.json'))
steps = [50, 100, 200, 500, 1000]
ddpm = [c2['DDPM'][str(s)] for s in steps]
ddim = [c2['DDIM'][str(s)] for s in steps]
fig, ax = plt.subplots(figsize=(COL_W, 2.45))
ax.plot(steps, ddpm, color=BLUE, marker='o', linestyle='-',
        ms=3.5, markeredgewidth=0.6, label='DDPM (stochastic)', clip_on=False)
ax.plot(steps, ddim, color=ORANGE, marker='s', linestyle='--',
        ms=3.5, markeredgewidth=0.6, label='DDIM (deterministic)', clip_on=False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Restoration steps at inference')
ax.set_ylabel('FID')
ax.xaxis.set_major_locator(FixedLocator(steps))
ax.set_xticklabels([str(s) for s in steps])
ax.yaxis.set_major_locator(FixedLocator([100, 200, 300, 500]))
ax.set_yticklabels(['100', '200', '300', '500'])
ax.minorticks_off()
ax.legend(frameon=False, handlelength=2.6, loc='upper right',
          borderaxespad=0.2, labelspacing=0.3)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig('figures/Fig7.png', bbox_inches='tight')
fig.savefig('figures/Fig7.eps', bbox_inches='tight')
plt.close(fig)

print('done')
