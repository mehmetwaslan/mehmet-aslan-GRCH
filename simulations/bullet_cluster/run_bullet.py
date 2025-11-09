# simulations/bullet_cluster/run_bullet.py
# GRCH Bullet Cluster Simulation - Full Proof Version
# Mehmet Aslan - November 9, 2025
# Fixed for GitHub: No lenstools, uses astropy only

import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
import os

# === 1. PARAMETERS ===
S0 = 0.950
G = 4.302e-6
M_cluster = 1.0e15
N_PER_CLUSTER = 100
m = M_cluster / N_PER_CLUSTER
epsilon = 30.0
dt = 0.8
n_steps = 50
sep = 2000.0
v_rel = 3000.0

# === 2. DRAG COEFF ===
rho_gas = 1.8e-27
v_shock = v_rel * 1e3
L_gas = 100e3 * 3.086e16
P_ram = rho_gas * v_shock**2
Sigma_gas = rho_gas * L_gas
a_drag = P_ram / Sigma_gas
kernel_radius = 120e3 * 3.086e16
drag_coeff = a_drag * kernel_radius / v_shock**2

print("=== GRCH BULLET CLUSTER ===")
print(f"S0: {S0:.3f}")
print(f"drag_coeff: {drag_coeff:.3f}")

# === 3. INITIAL CONDITIONS ===
np.random.seed(42)

def cluster_pos(center, sigma=180.0):
    return np.random.normal(center, sigma, (N_PER_CLUSTER, 3))

pos_b_main = cluster_pos([0, 0, 0])
pos_c_main = pos_b_main.copy()
pos_b_sub = cluster_pos([sep, 0, 0])
pos_c_sub = pos_b_sub.copy()

vel_b_main = np.random.normal(0, 60, (N_PER_CLUSTER, 3))
vel_b_sub = vel_b_main.copy() + [-v_rel, 0, 0]
vel_c_main = vel_b_main.copy()
vel_c_sub = vel_b_sub.copy()

pos = np.vstack([pos_b_main, pos_b_sub, pos_c_main, pos_c_sub])
vel = np.vstack([vel_b_main, vel_b_sub, vel_c_main, vel_c_sub])
masses = np.ones(len(pos)) * m
is_baryon = np.zeros(len(pos), dtype=bool)
is_baryon[:2*N_PER_CLUSTER] = True

# === 4. SPH DRAG ===
def apply_sph_drag(pos, vel, is_baryon, drag_coeff, kernel_radius=120.0):
    if not np.any(is_baryon):
        return vel
    n = sum(is_baryon)
    drag = np.zeros((n, 3))
    for i in range(n):
        r_vec = pos[is_baryon] - pos[is_baryon][i]
        r = np.linalg.norm(r_vec, axis=1)
        mask = (r > 1e-6) & (r < kernel_radius * 2)
        if np.sum(mask) > 0:
            dv_i = vel[is_baryon][mask] - vel[is_baryon][i]
            weight_i = np.exp(- (r[mask] / kernel_radius)**2)
            drag[i] = -drag_coeff * np.sum(weight_i[:, np.newaxis] * dv_i, axis=0)
    vel[is_baryon] += drag * dt
    return vel

# === 5. GRAVITY ===
def compute_gravity(pos, masses, is_baryon, S0):
    acc = np.zeros_like(pos)
    n = len(pos)
    for i in range(n):
        r_vec = pos - pos[i]
        r2 = np.sum(r_vec**2, axis=1) + epsilon**2
        r3 = r2 ** 1.5
        force_mag = G * masses[i] * masses / r3
        acc[i] = np.sum(force_mag[:, None] * r_vec, axis=0)
    acc[~is_baryon] *= S0
    return acc

# === 6. SIMULATION ===
print("Running simulation...", end="")
for step in range(n_steps):
    if step % 10 == 0:
        print(".", end="", flush=True)
    acc = compute_gravity(pos, masses, is_baryon, S0)
    vel += acc * dt
    vel = apply_sph_drag(pos, vel, is_baryon, drag_coeff)
    pos += vel * dt
print("\nSimulation complete.")

# === 7. OFFSET ===
gas_sub_center = np.mean(pos[N_PER_CLUSTER:2*N_PER_CLUSTER], axis=0)
curv_sub_center = np.mean(pos[3*N_PER_CLUSTER:4*N_PER_CLUSTER], axis=0)
offset_kpc = np.linalg.norm(gas_sub_center - curv_sub_center)

print(f"\nFinal Offset: {offset_kpc:.1f} kpc")
print("Target (paper): ~260 kpc")

# === 8. PLOT ===
os.makedirs('../../results', exist_ok=True)

# Simple convergence map
def make_convergence_map(pos, masses, box=6000, res=512):
    x, y = pos[:, 0], pos[:, 1]
    xi = np.linspace(-box/2, box/2, res)
    yi = np.linspace(-box/2, box/2, res)
    X, Y = np.meshgrid(xi, yi)
    grid = np.zeros((res, res))
    for px, py, m in zip(x, y, masses):
        i = int((px + box/2) / box * res)
        j = int((py + box/2) / box * res)
        if 0 <= i < res and 0 <= j < res:
            grid[j, i] += m
    Sigma_crit = 3.6e12
    kappa = grid / Sigma_crit
    kernel = Gaussian2DKernel(2.0)
    kappa_smooth = convolve(kappa, kernel)
    return X, Y, kappa_smooth

X, Y, kappa = make_convergence_map(pos, masses)

plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, kappa, cmap='viridis')
plt.colorbar(label='κ')
plt.scatter(gas_sub_center[0], gas_sub_center[1], c='red', s=100, label='Gas')
plt.scatter(curv_sub_center[0], curv_sub_center[1], c='cyan', s=100, label='Curvature')
plt.plot([gas_sub_center[0], curv_sub_center[0]], [gas_sub_center[1], curv_sub_center[1]], 'w--', lw=2)
plt.xlim(-2000, 2000)
plt.ylim(-1500, 1500)
plt.xlabel('X [kpc]')
plt.ylabel('Y [kpc]')
plt.title(f'GRCH Bullet Cluster: Offset = {offset_kpc:.1f} kpc')
plt.legend()
plt.tight_layout()
plt.savefig('../../results/bullet_offset.png', dpi=150)
plt.close()

print("Plot saved: results/bullet_offset.png")
