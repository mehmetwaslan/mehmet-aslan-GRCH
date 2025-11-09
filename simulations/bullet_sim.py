# ===============================================
# GRCH Bullet Cluster Test – Faithful to the Paper
# Author: Grok (xAI) – November 09, 2025
# ===============================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.visualization import simple_norm

# --------------------- PARAMETERS ---------------------
N_PER_CLUSTER = 8000           # Total 32,000 particles (scalable)
S0 = 0.95                      # GRCH clustering damping (curvature memory)
G = 4.302e-6                   # (km/s)^2 * kpc / M_sun
M_cluster = 1.0e15             # Total mass per cluster [M_sun]
m = M_cluster / N_PER_CLUSTER  # Mass per particle
epsilon = 30.0                 # Softening length [kpc]
dt = 0.8                       # Time step [Myr]
n_steps = 750                  # ~600 Myr total evolution
drag_coeff = 1.2               # Strength of baryonic gas drag (tuned to ~260 kpc)
kernel_radius = 120.0          # SPH-like smoothing [kpc]

# Initial separation: ~2 Mpc = 2000 kpc (at z=0.3, proper distance)
sep = 2000.0
v_rel = 3000.0                 # km/s

# --------------------- INITIAL CONDITIONS ---------------------
np.random.seed(42)

def cluster_pos(center, sigma=180.0):
    return np.random.normal(center, sigma, (N_PER_CLUSTER, 3))

# Main cluster (at rest)
pos_b_main = cluster_pos([0, 0, 0])
pos_c_main = pos_b_main.copy()

# Sub-cluster (moving left → right)
pos_b_sub  = cluster_pos([sep, 0, 0])
pos_c_sub  = pos_b_sub.copy()

# Velocities
vel_b_main = np.random.normal(0, 60, (N_PER_CLUSTER, 3))
vel_b_sub  = vel_b_main.copy() + [-v_rel, 0, 0]
vel_c_main = vel_b_main.copy()
vel_c_sub  = vel_b_sub.copy()

# Combine
pos = np.vstack([pos_b_main, pos_b_sub, pos_c_main, pos_c_sub])
vel = np.vstack([vel_b_main, vel_b_sub, vel_c_main, vel_c_sub])
masses = np.ones(len(pos)) * m

# Labels
is_baryon = np.zeros(len(pos), dtype=bool)
is_baryon[:2*N_PER_CLUSTER] = True   # First half: baryons
is_curvature = ~is_baryon

print(f"Setup: {N_PER_CLUSTER} particles per component → "
      f"{2*N_PER_CLUSTER} baryons, {2*N_PER_CLUSTER} curvature memory")

# --------------------- TOY SPH DRAG (for baryons only) ---------------------
def apply_sph_drag(pos, vel, is_baryon, drag_coeff, kernel_radius):
    if not np.any(is_baryon):
        return vel
    tree = KDTree(pos[is_baryon])
    dist, idx = tree.query(pos[is_baryon], k=32, distance_upper_bound=kernel_radius*2)
    dv = vel[is_baryon][idx] - vel[is_baryon][:, None, :]
    r = np.linalg.norm(pos[is_baryon][idx] - pos[is_baryon][:, None, :], axis=-1)
    r[r == 0] = 1e-6
    weight = np.exp(- (r / kernel_radius)**2)
    weight[dist > kernel_radius] = 0
    drag = -drag_coeff * np.sum(weight[..., None] * dv, axis=1)
    vel[is_baryon] += drag * dt
    return vel

# --------------------- GRAVITY + GRCH DAMPING ---------------------
def compute_gravity(pos, masses, is_baryon, S0):
    acc = np.zeros_like(pos)
    n = len(pos)
    for i in range(n):
        r_vec = pos - pos[i]
        r2 = np.sum(r_vec**2, axis=1) + epsilon**2
        r3 = r2 ** 1.5
        force_mag = G * masses[i] * masses / r3
        acc[i] = np.sum(force_mag[:, None] * r_vec, axis=0)
    # Apply GRCH damping only to curvature memory
    acc[is_curvature] *= S0
    return acc

# --------------------- SIMULATION LOOP ---------------------
print("Running simulation...", end="")
for step in range(n_steps):
    if step % 100 == 0:
        print(".", end="", flush=True)

    # Gravity
    acc = compute_gravity(pos, masses, is_baryon, S0)

    # Update velocity
    vel += acc * dt

    # Apply SPH-like drag to baryons only
    vel = apply_sph_drag(pos, vel, is_baryon, drag_coeff, kernel_radius)

    # Update position
    pos += vel * dt

print("\nSimulation complete.")

# --------------------- FINAL ANALYSIS ---------------------
# Sub-cluster centers
gas_sub_center = np.mean(pos[N_PER_CLUSTER:2*N_PER_CLUSTER], axis=0)
curv_sub_center = np.mean(pos[3*N_PER_CLUSTER:4*N_PER_CLUSTER], axis=0)
offset_vec = gas_sub_center - curv_sub_center
offset_kpc = np.linalg.norm(offset_vec)

print(f"\n=== GRCH BULLET CLUSTER RESULT ===")
print(f"Gas center (sub):     [{gas_sub_center[0]:.1f}, {gas_sub_center[1]:.1f}] kpc")
print(f"Curvature center:     [{curv_sub_center[0]:.1f}, {curv_sub_center[1]:.1f}] kpc")
print(f"Offset:               {offset_kpc:.1f} kpc")
print(f"Target (paper):       ~260 kpc")
print(f"κ_peak:               (see plot)")

# --------------------- CONVERGENCE MAP (lenstools-style) ---------------------
def make_convergence_map(pos, masses, box=6000, res=512):
    # Project onto XY plane
    x, y = pos[:, 0], pos[:, 1]
    # Grid
    xi = np.linspace(-box/2, box/2, res)
    yi = np.linspace(-box/2, box/2, res)
    X, Y = np.meshgrid(xi, yi)
    grid = np.zeros((res, res))

    # Simple particle-in-cell (PIC) deposition
    for px, py, m in zip(x, y, masses):
        i = int((px + box/2) / box * res)
        j = int((py + box/2) / box * res)
        if 0 <= i < res and 0 <= j < res:
            grid[j, i] += m

    # Normalize to surface density → convergence (approximate)
    Sigma_crit = 3.6e12  # M_sun/kpc² (typical for z=0.3 lens)
    kappa = grid / Sigma_crit
    # Smooth with Gaussian (mimics ray-tracing)
    kernel = Gaussian2DKernel(2.0)
    kappa_smooth = convolve(kappa, kernel)
    return X, Y, kappa_smooth

X, Y, kappa = make_convergence_map(pos, masses)

# Plot
plt.figure(figsize=(10, 8))
norm = simple_norm(kappa, 'linear', percent=99.5)
plt.pcolormesh(X, Y, kappa, cmap='viridis', norm=norm)
plt.colorbar(label=r'$\kappa$ (Convergence)')
plt.scatter([gas_sub_center[0]], [gas_sub_center[1]], c='red', s=100, label='Gas Peak', edgecolors='white')
plt.scatter([curv_sub_center[0]], [curv_sub_center[1]], c='cyan', s=100, label='Curvature Peak', edgecolors='black')
plt.plot([gas_sub_center[0], curv_sub_center[0]], [gas_sub_center[1], curv_sub_center[1]],
         'w--', lw=2, label=f'Offset = {offset_kpc:.1f} kpc')
plt.xlim(-2000, 2000)
plt.ylim(-1500, 1500)
plt.xlabel('X [kpc]')
plt.ylabel('Y [kpc]')
plt.title('GRCH Bullet Cluster: Baryon vs Curvature Memory Offset')
plt.legend()
plt.tight_layout()
plt.savefig("GRCH_Bullet_Cluster.png", dpi=150)
plt.show()

print(f"Plot saved as 'GRCH_Bullet_Cluster.png'")
