"""
derive_parameters.py

Small helper script to document the phenomenological choice of S0 and τ
used in the GRCH model.

Nothing is derived from first principles here. We simply:

- Set S0 = 0.95  -> late-time residual clustering factor
- Define f_decoh = 1 - S0 = 0.05  -> interpreted as a "non-decohered" curvature fraction
- Set τ ≃ 2.5 / H0  -> memory relaxation timescale, a few Hubble times

The purpose of this script is to show the numerical values and a simple
illustrative mapping, not to provide a microscopic derivation.
"""

import math

# Phenomenological parameter choices
S0 = 0.95          # residual clustering factor today
H0 = 67.4          # km/s/Mpc, used only to express τ in H0 units
tau_over_H0 = 2.5  # τ ≃ 2.5 / H0

# Simple "quantum decoherence" interpretation mapping (illustrative only)
f_decoh = 1.0 - S0          # non-decohered curvature fraction ~5%
T_RH_val = 1e9              # GeV, just to set a scale
Gamma_val = -math.log(f_decoh) * T_RH_val  # example "decoherence rate"

print(f"S0 (chosen)         : {S0:.3f}")
print(f"f_decoh = 1 - S0    : {f_decoh:.3f}")
print(f"Example Γ (T_RH=1e9): {Gamma_val:.3e} (arbitrary mapping)")
print(f"τ ≃ {tau_over_H0:.1f} / H0")

print("\nNOTE: S0 and τ are phenomenological choices,")
print("      not derived from first principles in this script.")
