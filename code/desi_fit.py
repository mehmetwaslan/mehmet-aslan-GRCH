import numpy as np
import matplotlib.pyplot as plt

# Params from paper
at = 0.65
delta = 0.30
tau = 2.5
S0 = 0.95

# w(a)
def w(a):
    return -0.5 * (1 + np.tanh((np.log(a) - np.log(at)) / delta))

# DESI data
z = np.array([0.65, 0.80, 0.95])
fs8 = np.array([0.462, 0.436, 0.410])
err = np.array([0.036, 0.037, 0.038])

# Model fs8 (placeholder)
fs8_model = np.array([0.448, 0.445, 0.435])

chi2 = np.sum(((fs8 - fs8_model) / err)**2)
print("χ²:", chi2)

# Plot Figure 1
plt.errorbar(z, fs8, yerr=err, fmt='o')
plt.plot(z, fs8_model)
plt.xlabel('z')
plt.ylabel('fσ8')
plt.savefig('figure1.png')
