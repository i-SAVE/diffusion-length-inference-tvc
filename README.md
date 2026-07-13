# Diffusion Process Length and Inference Algorithm — Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21320561.svg)](https://doi.org/10.5281/zenodo.21320561)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-i--SAVE%2Fdiffusion--length--inference--tvc-blue?logo=github)](https://github.com/i-SAVE/diffusion-length-inference-tvc)

Code, configurations, seeds, metrics and figure-generation scripts for:

> I. A. Kistin, Sh. M. Isaev. *Diffusion Process Length and Inference Algorithm in
> Denoising-Diffusion Image Synthesis: A Controlled Empirical Study on MNIST and CIFAR-10.*
> Submitted to **The Visual Computer**.

The study asks how the **training-time length of the diffusion process** (`T`) and the
**inference-time sampler** (DDPM vs DDIM) affect generation quality under a fixed training
budget, with the **terminal signal retention held constant** across all values of `T` so that
process length is the only variable.

---

## TL;DR for a reviewer (no GPU needed, ~5 seconds)

Verify that every number printed in Tables 1–3 of the manuscript is exactly the number
produced by the released runs:

```bash
python scripts/reproduce_tables.py --check
```

This reads only the JSON metric files under `results/` and asserts them against the values
in the paper. Expected output ends with:

```
OK: every value in Tables 1-3 matches the released metric files.
```

---

## What is in this package

```
├── src/
│   ├── mnist_diffusion.py            model, matched-noise schedule, DDPM/DDIM samplers, FID
│   └── cifar_diffusion.py            same, for 32×32 RGB
├── scripts/
│   ├── run_mnist_sweep.py            → Table 1 / Fig. 3
│   ├── run_cifar_experiments.py      → Tables 2–3 / Figs. 4 and 7
│   ├── export_samples.py             → the generated images the FIDs are computed from
│   ├── reproduce_tables.py           → prints/verifies Tables 1–3 from the metric files
│   ├── make_figures.py               → regenerates Figs. 3, 4, 7 (EPS + 600 dpi PNG)
│   └── collect_env.py                → records exact versions/GPU into results/logs/
├── configs/                          every hyper-parameter, seed and metric setting (YAML)
├── notebooks/                        executed notebooks (with saved outputs) + sources
├── results/
│   ├── mnist/budget_sweep.json       Table 1 (per-budget, per-T, mean & s.d. over seeds)
│   ├── cifar/cifar_table1_*.json     Table 2
│   ├── cifar/cifar_table2_*.json     Table 3
│   └── logs/                         full Kaggle session logs of the reported runs
└── figures/                          Figs. 1-10 (Figs. 3/4/7 also as vector EPS)
```

## Figure / table → source mapping

| Item in the paper | Figure file(s) | Produced by |
|---|---|---|
| Fig. 1 (training loss, five MNIST models) | `figures/Fig1.png` | `notebooks/original_experiments_executed.ipynb`, Experiment 1 |
| Fig. 2 (MNIST samples across T) | `figures/Fig2.png` | `notebooks/original_experiments_executed.ipynb`, Experiment 1 |
| Table 1, Fig. 3 (MNIST FID vs T, 3 budgets × 3 seeds) | `figures/Fig3.png` / `.eps` | `scripts/run_mnist_sweep.py` → `results/mnist/budget_sweep.json` → `scripts/make_figures.py` |
| Table 2, Fig. 4 (CIFAR-10 FID vs T, 2 seeds) | `figures/Fig4.png` / `.eps` | `scripts/run_cifar_experiments.py` (Experiment A) → `results/cifar/cifar_table1_fid_variance.json` → `scripts/make_figures.py` |
| Fig. 5 (DDPM step reduction, MNIST) | `figures/Fig5.png` | `notebooks/original_experiments_executed.ipynb`, Experiment 2 |
| Fig. 6 (DDIM at 50 steps, MNIST) | `figures/Fig6.png` | `notebooks/original_experiments_executed.ipynb`, Experiment 2 |
| Table 3, Fig. 7 (DDPM vs DDIM across steps, CIFAR-10) | `figures/Fig7.png` / `.eps` | `scripts/run_cifar_experiments.py` (Experiment B) → `results/cifar/cifar_table2_sampler_fid.json` → `scripts/make_figures.py` |
| Fig. 8 (intermediate reverse-process states) | `figures/Fig8.png` | `notebooks/original_experiments_executed.ipynb`, Experiment 3 |
| Fig. 9 (UMAP projections, Section 7) | `figures/Fig9.png` | `notebooks/original_experiments_executed.ipynb`, UMAP |
| Fig. 10 (classifier-free guidance, Section 7) | `figures/Fig10.png` | `notebooks/original_experiments_executed.ipynb`, guidance |
| Table 4 (positioning) | — | narrative table, no computation |

Figs. 1, 2, 5, 6, 8, 9, 10 are the authors' original outputs (PNG only, as produced
for the manuscript). Figs. 3, 4 and 7 are regenerated from `results/` and are the
only ones also shipped as vector EPS, per the journal's artwork specification.

**Note on provenance (important).** `notebooks/original_experiments_executed.ipynb` is the
authors' original executed notebook, shipped **with its saved outputs** so that Figs. 1, 2, 5, 6,
8, 9 and 10 can be inspected without re-running anything. Those runs are **single-seed**: they
established the qualitative phenomena, but they are *not* the source of Tables 1-3. The
quantitative results in the paper come from the **seed-controlled re-runs**
(`scripts/run_mnist_sweep.py`, `scripts/run_cifar_experiments.py`), which repeat every
configuration over multiple seeds and report FID as mean ± s.d. The single-seed FID-versus-T
curve visible in that notebook suggested a minimum at T = 200; the multi-seed re-runs did not
confirm that ordering (see Table 1 of the paper). It is retained for transparency, clearly
marked as superseded, because it is exactly what motivated the seed-controlled protocol.

**Note on the conditional model (Section 7).** It is a *separate* diffusers
`UNet2DConditionModel` (cross-attention label conditioning) trained with the *standard linear
DDPM schedule*, not the matched-noise schedule of the main experiments — as stated in the paper.

**Note on implementation variants.** `notebooks/local_variants/` contains a diffusers+scipy
CIFAR-10 variant that was **not** used for the reported numbers; the reported CIFAR-10 numbers
come from the Kaggle-executed pure-PyTorch notebook (full session log in `results/logs/`).
See `notebooks/local_variants/README.txt`.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/collect_env.py          # writes results/logs/environment.json
```

The reported runs were executed on a single NVIDIA GPU (Kaggle runtime, CUDA). The full
session logs, including every per-seed FID as it was printed, are in `results/logs/`.

## Reproducing the experiments

MNIST — Table 1 / Fig. 3 (5 step counts × 3 seeds, evaluated at 10/30/60 epochs):

```bash
python scripts/run_mnist_sweep.py                  # full sweep
python scripts/run_mnist_sweep.py --smoke          # 1 seed, 1 epoch (sanity check)
```

CIFAR-10 — Tables 2–3 / Figs. 4 and 7:

```bash
python scripts/run_cifar_experiments.py            # LIGHT preset = the paper setting
python scripts/run_cifar_experiments.py --smoke    # fast sanity check
```

Figures (regenerates the exact EPS/PNG shipped in `figures/`):

```bash
python scripts/make_figures.py
```

## Generated samples used for the metrics

The image sets fed to the Inception network are **regenerated bit-for-bit from the fixed
seeds** rather than shipped as large binaries. Each call writes a raw `.npz` tensor, a PNG
contact grid, and a `manifest.json` entry recording the seed, `T`, sampler, step count and a
SHA-256 checksum of the produced files:

```bash
# unconditional samples behind Tables 1–2
python scripts/export_samples.py --dataset mnist --T 700  --seed 0 --n 3000
python scripts/export_samples.py --dataset cifar --T 500  --seed 0 --n 2000

# the DDPM-vs-DDIM comparison behind Table 3 (T = 1000 anchor model)
python scripts/export_samples.py --dataset cifar --anchor --sampler ddpm --steps 50
python scripts/export_samples.py --dataset cifar --anchor --sampler ddim --steps 50
```

## Experimental controls (why the comparison is clean)

* **Matched terminal noise.** For every `T`, the endpoint `β_T` of the linear schedule is
  found by 100 iterations of binary search (bracket `[1e-4, 0.999]`) so that the cumulative
  signal retention satisfies `ᾱ_T ≈ 1e-3`. Longer chains therefore do **not** inject more
  total noise: only the number of steps changes.
* **Seed control.** Every configuration is trained from an explicit seed
  (`0, 1, 2` on MNIST; `0, 1` on CIFAR-10) and every FID is reported as mean ± s.d. over
  seeds. On MNIST, cuDNN autotuning is disabled and deterministic kernels are enabled.
* **Identical budget across `T`.** The same network, optimiser, learning rate and batch size
  are used for every step count, so `T` is the only variable.
* **Step subsetting.** When sampling with fewer steps than `T`, the retained indices are a
  uniform grid over `{0, …, T−1}` (`torch.linspace`), traversed from noisiest to cleanest;
  DDPM and DDIM use the *same* index grid.

## Known caveats (also stated in the manuscript)

* MNIST FID uses 3,000 real/generated images, CIFAR-10 uses 2,000; MNIST trains in fp32
  while CIFAR-10 uses automatic mixed precision, and the two FID routines differ slightly
  (the CIFAR-10 one symmetrises the covariances and clips negative eigenvalues before taking
  the matrix square root). Absolute FID values are therefore compared **only within a
  dataset**, never across the two.
* The conditional/classifier-free-guidance experiment (Section 7 of the paper) is auxiliary.
  Its feature-space distance is a random-projection proxy, not an Inception FID, and is
  reported only qualitatively.

## Citing this package

Two DOIs are available, per [Zenodo's versioning guidance](https://help.zenodo.org/):

- **To cite the exact snapshot that produced the numbers in the paper** (recommended
  for reproducibility), use the version-specific DOI for release **v1.0.2**:
  <https://doi.org/10.5281/zenodo.21321805>
- **To cite the software project in general**, independent of version, use the
  concept DOI (always resolves to the latest release):
  <https://doi.org/10.5281/zenodo.21320561>

```bibtex
@software{kistin_isaev_2026_diffusion_length,
  author    = {Kistin, I. A. and Isaev, Sh. M.},
  title     = {Code and metrics for "Diffusion Process Length and Inference
               Algorithm in Denoising-Diffusion Image Synthesis: A Controlled
               Empirical Study on MNIST and CIFAR-10"},
  year      = {2026},
  version   = {v1.0.2},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21320561},
  url       = {https://doi.org/10.5281/zenodo.21320561}
}
```

## License

MIT (see `LICENSE`). MNIST and CIFAR-10 are public datasets and are downloaded by the scripts.
