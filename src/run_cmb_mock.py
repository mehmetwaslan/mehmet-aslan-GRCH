# run_cmb_mock.py
# Mock CMB TT spectrum: GRCH vs ΛCDM vs Planck

ell = np.array([30, 50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000])
Dl_lcdm  = np.array([1800, 1400, 900, 600, 400, 300, 250, 220, 200, 180, 170, 165])
Dl_grch  = Dl_lcdm * (1 - 0.015)  # 1.5% suppression

Dl_planck = np.array([1850, 1350, 880, 590, 395, 298, 248, 218, 198, 178, 168, 163])
err       = np.array([  50,   40,  30,  20,  15,  12,  10,   9,   8,   7,   6,   6])
...
plt.errorbar(ell, Dl_planck, yerr=err, fmt='ko', label='Planck 2018')
plt.plot(ell, Dl_lcdm, 'r--', label='ΛCDM')
plt.plot(ell, Dl_grch, 'b-',  label='GRCH')
