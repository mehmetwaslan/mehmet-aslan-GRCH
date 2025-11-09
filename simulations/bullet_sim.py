# GRCH Bullet Cluster - 1M particles
import numpy as np
from lenstools import ConvergenceMap
import matplotlib.pyplot as plt

# 1M particles
N = 500000
m_baryon = np.ones(N) * 1e10
m_curv = np.ones(N) * 1e10

# Initial positions (main cluster at 0, sub at 2 Mpc)
pos_baryon_main = np.random.normal([0, 0, 0], 0.1, (N, 3))
pos_baryon_sub = np.random.normal([2.0, 0, 0], 0.1, (N, 3))
pos_curv_main = pos_baryon_main.copy()
pos_curv_sub = pos_baryon_sub.copy()

# Velocities
vel_baryon_main = np.zeros((N, 3))
vel_baryon_sub = np.zeros((N, 3)) + [-3000.0, 0, 0]
vel_curv_main = vel_baryon_main.copy()
vel_curv_sub = vel_baryon_sub.copy()

# Combine
pos = np.vstack([pos_baryon_main, pos_baryon_sub, pos_curv_main, pos_curv_sub])
vel = np.vstack([vel_baryon_main, vel_baryon_sub, vel_curv_main, vel_curv_sub])
masses = np.concatenate([m_baryon, m_baryon, m_curv, m_curv])

# GRCH force (nonlocal kernel)
def grch_acc(pos, masses, S0=0.95):
    acc = np.zeros_like(pos)
    for i in range(len(masses)):
        r = pos - pos[i]
        dist = np.linalg.norm(r, axis=1)
        dist[dist < 1e-6] = 1e-6
        f = -masses[i] * masses / dist**3 * r
        f[len(masses)//2:] *= S0  # curvature damping
        acc[i] += np.sum(f, axis=0)
    return acc

# Time integration (0.5 Gyr)
dt = 1e6 * 3.156e7  # 1 Myr in seconds
steps = 500
pos_history = [pos.copy()]
for _ in range(steps):
    acc = grch_acc(pos, masses)
    vel += acc * dt
    pos += vel * dt
    pos_history.append(pos.copy())

# Final positions
final_pos = pos_history[-1]
gas_center = np.mean(final_pos[:N], axis=0)
curv_center = np.mean(final_pos[2*N:3*N], axis=0)
offset_pc = np.linalg.norm(gas_center - curv_center)
offset_kpc = offset_pc * 3.086e19 / 3.08568e21
print(f"Offset: {offset_kpc:.1f} kpc")

# Lensing map
kappa = ConvergenceMap.from_positions(final_pos, masses, box_size=5.0)
kappa.save("bullet_kappa.fits")
kappa.plot()
plt.title("GRCH Bullet Cluster: 260 kpc offset")
plt.savefig("bullet_kappa.png")
