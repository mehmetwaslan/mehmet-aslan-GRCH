import numpy as np

N = 100  # small for test
S0 = 0.95
G = 4.3e-9  # Mpc (km/s)^2 / Msun
m = 5e13  # Msun per particle to match total mass ~5e15 Msun
epsilon = 0.01  # softening length in Mpc

pos_baryon_main = np.random.normal([0, 0, 0], 0.1, (N, 3))
pos_baryon_sub = np.random.normal([2.0, 0, 0], 0.1, (N, 3))
pos_curv_main = pos_baryon_main.copy()
pos_curv_sub = pos_baryon_sub.copy()

vel_baryon_main = np.zeros((N, 3))
vel_baryon_sub = np.zeros((N, 3)) + [-3000.0, 0, 0]
vel_curv_main = vel_baryon_main.copy()
vel_curv_sub = vel_baryon_sub.copy()

pos = np.vstack([pos_baryon_main, pos_baryon_sub, pos_curv_main, pos_curv_sub])
vel = np.vstack([vel_baryon_main, vel_baryon_sub, vel_curv_main, vel_curv_sub])
masses = np.ones(len(pos)) * m

def grch_acc(pos, masses, S0):
    n = len(pos)
    acc = np.zeros_like(pos)
    for i in range(n):
        r = pos - pos[i]
        dist = np.linalg.norm(r, axis=1)
        dist = np.sqrt(dist**2 + epsilon**2)
        a_contrib = (G * masses / dist**3)[:, np.newaxis] * r
        acc[i] = np.sum(a_contrib, axis=0)
    acc[2*N:] *= S0
    return acc

# Time integration
time_unit = 978000  # Myr per time unit (978 Gyr = 978000 Myr)
dt_physical = 1  # Myr
dt = dt_physical / time_unit  # ~1e-6
steps = 500

for _ in range(steps):
    acc = grch_acc(pos, masses, S0)
    vel += acc * dt
    pos += vel * dt

# Final
final_pos = pos
gas_center = np.mean(final_pos[N:2*N], axis=0)
curv_center = np.mean(final_pos[3*N:4*N], axis=0)
offset = np.linalg.norm(gas_center - curv_center)
offset_kpc = offset * 1000  # since pos in Mpc, *1000 to kpc
print(f"Offset: {offset_kpc:.1f} kpc")
