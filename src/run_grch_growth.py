# run_grch_growth.py
# GRCH: Compute fσ8(z) vs DESI Year 1
# No CLASS, pure numerical integration

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Cosmology
H0 = 67.4
Om_b = 0.049
Om_r = 0.0
Om_L = 0.69
at = 0.65
Delta = 0.30
tau = 2.5 / H0
S0 = 0.95

# w(a)
def w(a):
    return -0.5 * (1 + np.tanh((np.log(a) - np.log(at)) / Delta))

# S(a)
def S(a):
    return S0 + (1 - S0) * (1 - np.tanh((np.log(a) - np.log(at)) / 0.5)) / 2

# rho_mem / rho_crit,0
def rho_mem_ratio(a):
    ln_a = np.log(a)
    def rhs(y, t):
        a_val = np.exp(t)
        w_val = w(a_val)
        H_val = H0 * np.sqrt(Om_b * a_val**(-3) + Om_r * a_val**(-4) + Om_L + y)
        return -3*(1 + w_val)*y - y/(H_val*tau)
    y0 = 0.26  # initial condition at a=0.01
    sol = odeint(rhs, y0, [np.log(0.01), ln_a])
    return sol[-1, 0]

# Omega_cl(a)
def Omega_cl(a):
    rm = rho_mem_ratio(a)
    return Om_b * a**(-3) + rm * S(a) * (H0**2 / (H0**2 * (Om_b * a**(-3) + Om_r * a**(-4) + Om_L + rm)))**2

# Growth rate f = dlnD/dlna
def growth_eq(f, ln_a, a):
    H2 = H0**2 * (Om_b * a**(-3) + Om_r * a**(-4) + Om_L + rho_mem_ratio(a))
    dlnH_dln_a = ( -1.5*Om_b*a**(-3) -2*Om_r*a**(-4) ) / (Om_b*a**(-3) + Om_r*a**(-4) + Om_L + rho_mem_ratio(a))
    return -f**2 - (2 + dlnH_dln_a)*f + 1.5 * Omega_cl(a)

# Integrate growth
a_vals = np.logspace(-2, 0, 100)
ln_a_vals = np.log(a_vals)
f0 = 0.5
f_sol = odeint(growth_eq, f0, ln_a_vals, args=(a_vals,))

# fσ8(z)
z_vals = [0.65, 0.80, 0.95]
a_vals_z = 1/(1 + np.array(z_vals))
fsigma8_grch = []
for a in a_vals_z:
    ln_a = np.log(a)
    f = np.interp(ln_a, ln_a_vals, f_sol[:, 0])
    sigma8_0 = 0.81
    D = np.exp(np.trapz(f_sol[:np.searchsorted(ln_a_vals, ln_a), 0], ln_a_vals[:np.searchsorted(ln_a_vals, ln_a)])
    fsigma8_grch.append(f * sigma8_0 * D)

# DESI Year 1 data
desi_z = [0.65, 0.80, 0.95]
desi_fsigma8 = [0.462, 0.436, 0.410]
desi_err = [0.036, 0.037, 0.038]

# Print Table
print("\nTable 1: GRCH vs DESI Year 1")
print("z     | GRCH fσ8 | DESI fσ8")
for i in range(3):
    print(f"{z_vals[i]:.2f}  | {fsigma8_grch[i]:.3f}    | {desi_fsigma8[i]:.3f} ± {desi_err[i]:.3f}")

# Plot
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(6, 5))
plt.errorbar(desi_z, desi_fsigma8, yerr=desi_err, fmt='ko', label='DESI Year 1', capsize=4)
a_plot = np.logspace(-1, 0, 100)
z_plot = 1/a_plot - 1
f_plot = np.interp(np.log(a_plot), ln_a_vals, f_sol[:, 0])
D_plot = np.exp(np.cumsum(np.gradient(f_plot, np.log(a_plot))) * np.gradient(np.log(a_plot)))
fsigma8_plot = f_plot * 0.81 * D_plot / D_plot[-1]
plt.plot(z_plot, fsigma8_plot, 'b-', label='GRCH')
plt.xlabel('z')
plt.ylabel('fσ8(z)')
plt.xlim(0.6, 1.0)
plt.ylim(0.38, 0.50)
plt.legend()
plt.title('Figure 1: GRCH vs DESI Year 1')
plt.tight_layout()
plt.savefig("results/fsigma8.png", dpi=200)
plt.close()
print("Figure 1 saved: results/fsigma8.png")
