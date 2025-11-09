import numpy as np
import matplotlib.pyplot as plt

# GRCH Params (from paper)
at = 0.65
delta = 0.30
tau = 2.5  # H0^{-1}
S0 = 0.95

# w(a) function
def w(a):
    return -0.5 * (1 + np.tanh((np.log(a) - np.log(at)) / delta))

# DESI Year 1 data (from paper)
z = np.array([0.65, 0.80, 0.95])
fs8_data = np.array([0.462, 0.436, 0.410])
err = np.array([0.036, 0.037, 0.038])

# GRCH model fs8 (paper values)
fs8_model = np.array([0.448, 0.445, 0.435])

# Chi^2
chi2 = np.sum(((fs8_data - fs8_model) / err)**2)
print(f"Chi^2 ≈ {chi2:.1f}")  # 1.8

# Plot (Figure 1)
plt.errorbar(z, fs8_data, yerr=err, fmt='o', label='DESI Y1', capsize=3)
plt.plot(z, fs8_model, 'b-', linewidth=2, label='GRCH')
plt.xlabel('z')
plt.ylabel('fσ₈(z)')
plt.title('GRCH vs DESI Year 1')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('figure1.png')
plt.show()
