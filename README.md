# Learning the Precision on Fashion-MNIST and EMNIST

Experiments for the learned-precision extension of Possibilistic Variational Inference.
While the baseline CBOpt optimizers fix the candidate precision vartheta by hand,
here dispersion is learned variationally under an extremal-volume penalty (scale zeta = d/2):
isotropic and diagonal precisions via a damped Picard fixed point with anti-collapse and
positivity caps, and low-rank-plus-diagonal (LRD) and layer-wise block-diagonal (BD)
precisions via closed-form, sign-asymmetric safeguards. Sharpness-Aware Minimization (SAM)
is connected to the lower CBO (s = -1) via an indicator candidate and extended through
a Gaussian relaxation to extract an adaptive perturbation radius.

Benchmark protocol matches reproduction: LeNet-5 on Fashion-MNIST (90/10 train/val split,
54k/6k, 10k test), EMNIST-Letters as OOD, batch size 128, linear warmup (5 epochs)
with cosine annealing. Experimental setup:
- Benchmark: uCBOpt iso/diag/BD, s = +1, 3 seeds x 100 epochs (main evaluation table).
- Geometry check: lCBOpt LRD/BD, s = -1, 1 seed x 50 epochs (flip rate, feasibility diagnostics)
- SAM comparison: 6 independent runs (3 possibilistic lCBOpt-iso s = -1 with r in {1, 0.95, 1.05} x r_nom, 3 standard SAM with rho in {1, 0.95, 1.05} x rho_bar_nom), 1 seed x 50 epochs.
H_s > 0 is reported strictly before the cap (the after-cap level is guaranteed by construction).


## Structure

```
src/
  data.py         -- Fashion-MNIST and EMNIST data loaders
  model.py        -- LeNet-5 architecture (44,426 parameters)
  optimizers.py   -- CBOptIso / CBOptDiag / CBOptLRD / CBOptBlock / SAM
  train.py        -- training loop, per-step diagnostics hook, SAM two-pass step
  diagnostics.py  -- per-step before-cap H_s > 0, L_Phi, and feasibility-by-s_eff metrics
  evaluate.py     -- Acc, NLL, ECE, FPR@95, and AUROC metrics
  toy.py          -- closed-form isotropic quadratic verification
checkpoints/      -- model checkpoints, history CSVs, and per-step trajectories
results/          -- evaluation metric tables (spread_summary.csv, spread_raw.csv)
figs/             -- exported diagnostic and landscape figures
xp_spread.ipynb   -- main experiment notebook (Sections 0-4)
requirements.txt
```


## Setup

The experiment notebook is configured for Google Colab (T4 GPU runtime).
To run on a local machine with a CUDA-enabled GPU instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Usage

Open xp_spread.ipynb and execute cells in order:

- Google Colab: Open the notebook directly on Colab and run all cells sequentially.
- Local environment: Activate the virtual environment, then skip Cell 1 (Google Drive mount) before executing the rest of the notebook.

Notebook outline:
- TOY (closed form, no training): Closed-form checks on the isotropic quadratic (regularity and continuity of theta_s(vartheta), monotonicity repair via log-volume penalty, sensitivity to penalty scale zeta = d/2, collinearity of eps*_SAM vs u*_soft), and dimension sweep (LRD < BD < iso < diag).
- Benchmark training: uCBOpt-iso, uCBOpt-diag, and uCBOpt-BD with s = +1 (3 seeds x 100 epochs).
- Benchmark result: final comparison table against fixed uCBOpt, AdamW, and IVON baselines.
- H_s > 0 before the cap: fraction of steps where H_s > 0 holds naturally, before any cap, for uCBOpt-iso and uCBOpt-diag.
- Flip check: lCBOpt-LRD and lCBOpt-BD with s = -1 (flip rate and feasibility failure tracking).
- SAM: learned radius (adaptive) vs fixed radius, 6 runs logging adaptive trajectory rho*_t and effective radius rho_bar_nom.
- Picard contraction constant: L_Phi tracking for s = +1 and s = -1.


## Results (mean +/- std, 3 seeds -- Benchmark)

Single deterministic forward pass for all CBOpt variants and AdamW, 64-sample Monte-Carlo ensemble for IVON.
lCBOpt-LRD / lCBOpt-BD (s = -1) and SAM runs are evaluated separately as diagnostics, not in this table.
uCBOpt-BD aggregates the two non-diverging seeds (seed 2 diverged at epoch 16).

| Optimizer | Acc | NLL | ECE | FPR@95 | AUROC |
|---|---|---|---|---|---|
| uCBOpt-iso (s=+1, learned vartheta) | 0.912 +/- 0.001 | 0.257 +/- 0.002 | 0.022 +/- 0.001 | 0.263 +/- 0.015 | 0.793 +/- 0.018 |
| uCBOpt-diag (s=+1, learned bold vartheta) | 0.915 +/- 0.001 | 0.252 +/- 0.001 | 0.020 +/- 0.002 | 0.292 +/- 0.018 | 0.758 +/- 0.018 |
| uCBOpt-BD (s=+1, learned S) | 0.908 +/- 0.002 | 0.256 +/- 0.001 | 0.014 +/- 0.001 | 0.289 +/- 0.017 | 0.755 +/- 0.018 |
| uCBOpt (fixed vartheta) | 0.909 +/- 0.002 | 0.272 +/- 0.004 | 0.025 +/- 0.001 | 0.263 +/- 0.005 | 0.795 +/- 0.005 |
| IVON | 0.910 +/- 0.002 | 0.258 +/- 0.003 | 0.019 +/- 0.001 | 0.241 +/- 0.009 | 0.812 +/- 0.012 |
| AdamW | 0.896 +/- 0.003 | 0.296 +/- 0.005 | 0.022 +/- 0.006 | 0.335 +/- 0.005 | 0.722 +/- 0.016 |