# run_cmb_mock.py
# Toy mock CMB TT spectrum for GRCH vs LCDM vs approximate Planck points
# -----------------------------------------------------------------------------
# IMPORTANT:
# This script is a phenomenological illustration ONLY.
# It does NOT perform a physical Boltzmann calculation.
# No CLASS or CAMB computations are involved here.
# -----------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Multipoles
# --------------------------------------------------------------------------
ell = np.array([30, 50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000])

# --------------------------------------------------------------------------
# Approximate LCDM TT values (toy numbers, in μK^2)
# --------------------------------------------------------------------------
Dl_lcdm = np.array([1800, 1400, 900, 600, 400, 300, 250, 220, 200, 180, 170, 165])

# --------------------------------------------------------------------------
# GRCH mock: small percent-level suppression of low-ℓ power
# Here we take a uniform 1.5% suppression as a simple toy example.
# --------------------------------------------------------------------------
suppression_factor = 0.985  # 1.5% suppression
Dl_grch = Dl_lcdm * suppression_factor

# --------------------------------------------------------------------------
# Approximate "Planck-like" TT points and error bars (toy numbers)
# --------------------------------------------------------------------------
Dl_planck = np.array([1850, 1350, 880, 590, 395, 298, 248, 218, 198, 178, 168, 163])
err = np.array([50, 40, 30, 20, 15, 12, 10, 9, 8, 7, 6, 6])

# --------------------------------------------------------------------------
# Simple toy chi-square check: how far is the GRCH mock from these points?
# This chi-square is purely illustrative and should NOT be interpreted as
# a real likelihood analysis.
# --------------------------------------------------------------------------
chi2 = np.sum(((Dl_grch - Dl_planck) / err) ** 2)
print(f"Toy χ² (GRCH mock vs approximate Planck points) ≈ {chi2:.2f}")

# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(7, 5))
plt.errorbar(ell, Dl_planck, yerr=err, fmt='ko', label='Planck (approx.)')
plt.plot(ell, Dl_lcdm, 'r--', label='LCDM mock')
plt.plot(ell, Dl_grch, 'b-', label='GRCH mock (1.5% suppression)')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell\ [\mu{\rm K}^2]$')
plt.title('Toy CMB TT: GRCH vs LCDM vs approximate Planck')
plt.legend()
plt.tight_layout()
plt.savefig("results/cmb_mock.png", dpi=200)
plt.close()

print("Figure saved: results/cmb_mock.png")
