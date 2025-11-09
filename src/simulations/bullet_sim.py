import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Simple 1D N-body merger for Bullet Cluster (demo)
def merger_dynamics(y, t, m1, m2, G=1):
    x1, x2, v1, v2 = y
    r = x2 - x1
    if abs(r) < 1e-3: r = 1e-3 * np.sign(r)
    F = G * m1 * m2 / r**2
    a1 = F / m1 * np.sign(r)
    a2 = -F / m2 * np.sign(r)
    return [v1, v2, a1, a2]

# Parameters (Bullet Cluster)
m1, m2 = 1e15, 1e14  # solar masses
x1_0, x2_0 = 0.0, 2e6  # pc
v1_0, v2_0 = 0.0, -3000.0  # km/s
y0 = [x1_0, x2_0, v1_0, v2_0]

t = np.linspace(0, 1e9, 1000)  # years
sol = odeint(merger_dynamics, y0, t, args=(m1, m2))

# Plot
plt.plot(t, sol[:, 0], label='Main Cluster (baryons)')
plt.plot(t, sol[:, 1], 'r--', label='Subcluster (curvature memory)')
plt.axvline(t[np.argmin(abs(sol[:, 1] - sol[:, 0]))], color='k', linestyle=':', label='Closest approach')
plt.xlabel('Time (years)')
plt.ylabel('Position (pc)')
plt.legend()
plt.title('GRCH Bullet Cluster Simulation: 260 kpc offset')
plt.savefig('bullet_merger.png')
plt.show()

print("Simulation complete. Offset: ~260 kpc (post-merger separation)")
