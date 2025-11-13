"""
derive_parameters.py

GRCH için kullanılan S0 ve τ değerlerinin GERÇEKÇİ ama fenomenolojik
seçimini açıklayan minik yardımcı script.

Burada hiçbir şeyi "mikrofizikten türetmiyoruz". Sadece:

- S0 = 0.95  -> late-time residual clustering factor
- f_decoh = 1 - S0 = 0.05  -> "decohere olmamış" curvature fraksiyonu yorumu
- τ = 2.5 / H0  -> memory relaxation süresi, H0 ölçeğinin birkaç katı

Bu script'in amacı, makaledeki sayısal değerlerle bu sezgisel resim
arasındaki bağlantıyı göstermek; not a first-principles derivation.
"""

import math

# Seçilen phenomenolojik parametreler
S0 = 0.95          # residual clustering factor today
H0 = 67.4          # km/s/Mpc, sadece τ'nin ifadesi için
tau_over_H0 = 2.5  # τ ≃ 2.5 / H0

# "Quantum decoherence" yorumu için basit mapping
f_decoh = 1.0 - S0          # decohere olmamış curvature fraksiyonu ~5%
T_RH_val = 1e9              # GeV, sadece ölçek fikri vermek için
Gamma_val = -math.log(f_decoh) * T_RH_val  # örnek bir "decoherence rate"

print(f"S0 (chosen)         : {S0:.3f}")
print(f"f_decoh = 1 - S0    : {f_decoh:.3f}")
print(f"Example Γ (T_RH=1e9): {Gamma_val:.3e} (arbitrary mapping)")
print(f"τ ≃ {tau_over_H0:.1f} / H0")

print("\nNOTE: S0 and τ are phenomenological choices,")
print("      not derived from first principles in this script.")
