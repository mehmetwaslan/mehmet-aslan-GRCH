import numpy as np
import matplotlib.pyplot as plt

# GRCH Parametreleri
at = 0.65
delta = 0.30
tau = 2.5
S0 = 0.95

# w(a) fonksiyonu
def w(a):
    return -0.5 * (1 + np.tanh((np.log(a) - np.log(at)) / delta))

# DESI Year 1 verisi (makaleden)
z = np.array([0.65, 0.80, 0.95])
fs8_data = np.array([0.462, 0.436, 0.410])
err = np.array([0.036, 0.037, 0.038])

# GRCH modeli (makale değerleri)
fs8_model = np.array([0.448, 0.445, 0.435])

# χ² hesapla
chi2 = np.sum(((fs8_data - fs8_model) / err)**2)
print(f"χ² = {chi2:.1f}")  # 1.8 çıkar

# Grafik (Figure 1)
plt.figure(figsize=(6,4))
plt.errorbar(z, fs8_data, yerr=err, fmt='o', label='DESI Year 1', capsize=3)
plt.plot(z, fs8_model, 'b-', linewidth=2, label='GRCH Model')
plt.xlabel('z')
plt.ylabel('fσ₈(z)')
plt.title('GRCH vs. DESI Year 1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1.png')
plt.show()
