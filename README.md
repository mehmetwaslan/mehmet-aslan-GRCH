# GRCH – Geometric Residual Curvature Hypothesis

Supplementary code for the manuscript

**“The Geometric Residual Curvature Hypothesis (GRCH):  
A Non-Dark-Matter Explanation of Cosmic Structure Growth”**

This repository contains simple numerical experiments and toy models that
illustrate the qualitative phenomenology discussed in the paper.  
They are **not** full Boltzmann or hydrodynamical simulations and are not
intended to replace standard cosmological codes such as CLASS or CAMB.

---

## Code layout (`src/`)

### 1. `run_bullet.py` – Toy Bullet Cluster N-body model

A simplified N-body + drag experiment showing how a collisionless curvature
component can separate from baryonic gas during a high-velocity cluster
merger. This is **not** a hydrodynamical simulation; it is only a conceptual
toy model demonstrating the expected qualitative offset between gas and
“curvature mass”.

**Main outputs**

- `results/bullet_offset.png`  
  – projected mass map with gas vs curvature centroids and measured offset  
- Console output: offset distance and peak convergence `κ_peak`.

---

### 2. `run_grch_growth.py` – GRCH growth ODE solver

Numerically solves the modified growth equation

\[
\frac{df}{d\ln a} + f^2 + \left(2 + \frac{d\ln H}{d\ln a}\right) f
= \frac{3}{2}\,\Omega_{\rm cl}(a),
\]

with a GRCH-inspired clustering fraction and background, and computes

\[
f\sigma_8(z)
\]

at the three DESI Year-1 redshifts. The script then compares the model
values to the DESI measurements and prints a simple χ².

**Main outputs**

- `results/fsigma8_grch_vs_desi.png`  
  – model curve plus DESI Y1 points  
- Console output, e.g.

  ```text
  z = 0.65 | model fσ8 = ... | DESI = ... ± ...
  z = 0.80 | ...
  z = 0.95 | ...
  χ² (3 pts) ≈ ...
