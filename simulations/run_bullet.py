# run_bullet.py
# GRCH Bullet Cluster Simulation - 260 kpc offset, no lenstools
# Mehmet Aslan - November 12, 2025

import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
import os

# Parameters
S0 = 0.950
G = 4.302e-6
M_cluster = 1.0e15
N_PER_CLUSTER = 100
m = M_cluster / N_PER_CLUSTER
epsilon = 30.0
dt = 0.05
n_steps = 28
sep = 2000.0
v_rel = 3000.0
drag_coeff = 2.8
kernel_radius = 120.0

print("=== GRCH BULLET CLUSTER SIMULATION ===")
print(f"S0: {S0:.3f} | drag_coeff: {drag_coeff:.1f} | time: {n_steps*dt:.1f} Gyr")

# Initial conditions
np.random.seed(42)
def cluster_pos(center, sigma=180.0):
    return np.random.normal(center, sigma, (N_PER_CLUSTER, 3))

pos_b_main = cluster_pos([0, 0, 0])
pos_c_main = pos_b_main.copy()
pos_b_sub = cluster_pos([sep, 0, 0])
pos_c_sub = pos_b_sub.copy()

vel_b_main = np.random.normal(0, 600, (N_PER_CLUSTER, 3))
vel_b_sub = vel_b_main.copy() + [-v_rel, 0, 0]
vel_c_main = vel_b_main.copy()
vel_c_sub = vel_b_sub.copy()

pos = np.vstack([pos_b_main, pos_b_sub, pos_c_main, pos_c_sub])
vel = np.vstack([vel_b_main, vel_b_sub, vel_c_main, vel_c_sub])
masses = np.ones(len(pos)) * m
is_baryon = np.zeros(len(pos), dtype=bool)
is_baryon[:2*N_PER_CLUSTER] = True

# Drag
def apply_sph_drag(pos, vel, is_baryon, drag_coeff, kernel_radius):
    if not np.any(is_baryon): return vel
    drag = np.zeros_like(vel)
    baryon_idx = np.where(is_baryon)[0]
    for i in baryon_idx:
        r_vec = pos[is_baryon] - pos[i]
        r = np.linalg.norm(r_vec, axis=1)
        mask = (r > 1e-6) & (r < kernel_radius * 2)
        if np.sum(mask) == 0: continue
        dv = vel[is_baryon][mask] - vel[i]
        w = np.exp(-(r[mask] / kernel_radius)**2)
        drag[i] = -drag_coeff * np.sum(w[:, None] * dv, axis=0) / np.sum(w)
    vel += drag * dt
    return vel

# Gravity
def compute_gravity(pos, masses, is_baryon, S0, epsilon):
    acc = np.zeros_like(pos)
    for i in range(len(pos)):
        r_vec = pos - pos[i]
        r2 = np.sum(r_vec**2, axis=1) + epsilon**2
        mask = r2 > 0
        r3 = r2[mask]**1.5
        force = G * masses[mask] / r3
        acc[i] = np.sum(force[:, None] * r_vec[mask], axis=0)
    acc[~is_baryon] *= S0
    return acc

# Simulation
print("Running simulation...", end="")
for step in range(n_steps):
    if step % 7 == 0: print(".", end="", flush=True)
    acc = compute_gravity(pos, masses, is_baryon, S0, epsilon)
    vel += acc * dt
    vel = apply_sph_drag(pos, vel, is_baryon, drag_coeff, kernel_radius)
    pos += vel * dt
print("\nSimulation complete.")

# Offset
gas_center = np.mean(pos[N_PER_CLUSTER:2*N_PER_CLUSTER], axis=0)
curv_center = np.mean(pos[3*N_PER_CLUSTER:], axis=0)
offset = np.linalg.norm(gas_center - curv_center)
print(f"Final Offset: {offset:.1f} kpc")

# Convergence map
def make_kappa(pos, masses, box=6000, res=512):
    x, y = pos[:, 0], pos[:, 1]
    xi = np.linspace(-box/2, box/2, res)
    yi = np.linspace(-box/2, box/2, res)
    X, Y = np.meshgrid(xi, yi)
    grid = np.zeros((res, res))
    for px, py, pm in zip(x, y, masses):
        i = int((px + box/2) / box * (res - 1))
        j = int((py + box/2) / box * (res - 1))
        if 0 <= i < res and 0 <= j < res:
            grid[j, i] += pm
    Sigma_crit = 5.0e17  # Msun/kpc^2
    kappa = grid / Sigma_crit
    kernel = Gaussian2DKernel(1.5)
    kappa_smooth = convolve(kappa, kernel, boundary='extend')
    return X, Y, kappa_smooth

X, Y, kappa = make_kappa(pos, masses)
kappa_peak = np.max(kappa)
print(f"Convergence peak (κ): {kappa_peak:.2f}")

# Plot
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, kappa, cmap="viridis", vmin=0, vmax=0.5, shading='auto')
plt.colorbar(label="κ")
plt.scatter(gas_center[0], gas_center[1], c='red', s=150, label='Gas', edgecolor='white')
plt.scatter(curv_center[0], curv_center[1], c='cyan', s=150, label='Curvature', edgecolor='white')
plt.plot([gas_center[0], curv_center[0]], [gas_center[1], curv_center[1]], 'w--', lw=3)
plt.xlim(-2000, 2000)
plt.ylim(-1500, 1500)
plt.xlabel("X [kpc]")
plt.ylabel("Y [kpc]")
plt.title(f"GRCH Bullet Cluster\nOffset = {offset:.1f} kpc | κ_peak = {kappa_peak:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("results/bullet_offset.png", dpi=200)
plt.close()

print("Plot saved: results/bullet_offset.png")
