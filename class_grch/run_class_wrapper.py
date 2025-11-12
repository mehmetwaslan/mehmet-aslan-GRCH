# src/run_class_wrapper.py
# Run CLASS and plot results
from class_grch.grch import run_class
import numpy as np
import matplotlib.pyplot as plt
import os

run_class()

# Load CLASS output
cl = np.loadtxt("results/class_output/grch_cl.dat")
ell = cl[:, 0]
Dl_tt = cl[:, 1] * ell * (ell + 1) / (2 * np.pi)

# Plot
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(7, 5))
plt.plot(ell, Dl_tt, 'b-', label='GRCH (CLASS)')
plt.xlabel('ℓ')
plt.ylabel('D_ℓ^{TT} [μK²]')
plt.xlim(30, 2000)
plt.ylim(0, 6000)
plt.legend()
plt.title('Figure 2: CMB TT Spectrum (CLASS)')
plt.tight_layout()
plt.savefig("results/cmb_tt_class.png", dpi=200)
plt.close()
print("Saved: results/cmb_tt_class.png")
