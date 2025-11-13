# run_grch_growth.py
# GRCH: Compute fσ8(z) vs DESI Year 1
# Pure numerical integration of the growth equation (no CLASS)

import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ----------------------------
# Cosmology / GRCH parameters
# ----------------------------
H0 = 67.4
Om_b = 0.049
Om_mem0 = 0.26           # effective "memory" density today
Om_m = Om_b + Om_mem0    # total clustering matter in background
Om_r = 0.0
Om_L = 1.0 - Om_m - Om_r

at = 0.65                # transition scale factor
S0 = 0.95                # residual clustering factor today
sigma8_0 = 0.81          # present-day σ8 normalization


# ----------------------------
# Helper functions
# ----------------------------
def S_of_a(a: float) -> float:
    """Scale-dependent clustering suppression S(a)."""
    x = math.log(a) - math.log(at)
    return S0 + (1.0 - S0) * (1.0 - math.tanh(x / 0.5)) / 2.0


def E2_of_a(a: float) -> float:
    """E^2(a) = H^2(a)/H0^2."""
    return Om_m * a**-3 + Om_r * a**-4 + Om_L


def dlnH_dlnA(a: float) -> float:
    """Derivative d ln H / d ln a."""
    E2 = E2_of_a(a)
    dE2_dlnA = -3.0 * Om_m * a**-3 - 4.0 * Om_r * a**-4
    return 0.5 * dE2_dlnA / E2


def Omega_cl(a: float) -> float:
    """Effective clustering fraction Ω_cl(a)."""
    E2 = E2_of_a(a)
    return (Om_b + S_of_a(a) * Om_mem0) * a**-3 / E2


def df_dlnA(ln_a: float, f: float) -> float:
    """Growth equation df/d ln a."""
    a = math.exp(ln_a)
    return -f**2 - (2.0 + dlnH_dlnA(a)) * f + 1.5 * Omega_cl(a)


# ----------------------------
# Solve f(ln a) with RK4
# ----------------------------
a_min = 1e-3
a_max = 1.0
n_steps = 2000

ln_a_vals = np.linspace(math.log(a_min), math.log(a_max), n_steps)
f_vals = np.zeros(n_steps)

# Initial condition: deep in matter domination, f ≈ 1
f_vals[0] = 1.0

for i in range(1, n_steps):
    h = ln_a_vals[i] - ln_a_vals[i - 1]
    x = ln_a_vals[i - 1]
    f = f_vals[i - 1]

    k1 = df_dlnA(x, f)
    k2 = df_dlnA(x + 0.5 * h, f + 0.5 * h * k1)
    k3 = df_dlnA(x + 0.5 * h, f + 0.5 * h * k2)
    k4 = df_dlnA(x + h, f + h * k3)

    f_vals[i] = f + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ----------------------------
# Compute D(a) and fsigma8(a)
# ----------------------------
D_ln = np.zeros(n_steps)
for i in range(1, n_steps):
    h = ln_a_vals[i] - ln_a_vals[i - 1]
    D_ln[i] = D_ln[i - 1] + f_vals[i - 1] * h

D_vals = np.exp(D_ln)
D_vals /= D_vals[-1]  # normalize so D(a=1) = 1

fs8_vals = f_vals * sigma8_0 * D_vals


def fs8_at_z(z: float) -> float:
    a = 1.0 / (1.0 + z)
    ln_a = math.log(a)
    f = np.interp(ln_a, ln_a_vals, f_vals)
    D = np.interp(ln_a, ln_a_vals, D_vals)
    return f * sigma8_0 * D


# ----------------------------
# DESI Year 1 comparison
# ----------------------------
z_desi = np.array([0.65, 0.80, 0.95])
fs8_desi = np.array([0.462, 0.436, 0.410])
err_desi = np.array([0.036, 0.037, 0.038])

chi2 = 0.0
for z, d, e in zip(z_desi, fs8_desi, err_desi):
    m = fs8_at_z(z)
    chi2 += ((m - d) / e)**2
    print(f"z = {z:.2f} | model fσ8 = {m:.3f} | DESI = {d:.3f} ± {e:.3f}")

print(f"χ² (3 pts) ≈ {chi2:.3f}")


# ----------------------------
# Plot fσ8(z)
# ----------------------------
z_plot = np.linspace(0.6, 1.0, 200)
fs8_plot = np.array([fs8_at_z(z) for z in z_plot])

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(7, 5))
plt.plot(z_plot, fs8_plot, 'b-', label='GRCH (growth ODE)')
plt.errorbar(z_desi, fs8_desi, yerr=err_desi, fmt='o', color='black', label='DESI Year 1')
plt.xlabel('z')
plt.ylabel(r'$f\sigma_8(z)$')
plt.xlim(0.6, 1.0)
plt.ylim(0.38, 0.50)
plt.legend()
plt.title('GRCH growth vs DESI Year 1 (toy ODE model)')
plt.tight_layout()
plt.savefig("results/fsigma8.png", dpi=200)
plt.close()

print("Figure saved: results/fsigma8.png")
