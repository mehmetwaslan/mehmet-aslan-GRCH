
---

## **2. src/class_mod.c** (CLASS için örnek modifikasyon)

```c
// GRCH custom fluid for CLASS
// Add to background.c or perturbations.c

#include "common.h"

int background_grch(
     struct background *pba,
     double a,
     double *rho_mem,
     double *p_mem,
     double *w_mem
     ) {
  double a_t = 0.65;
  double Delta = 0.30;
  double tau = 2.5 / pba->H0;  // in conformal time units
  double S0 = 0.95;

  // w(a) from tanh
  *w_mem = -0.5 * (1.0 + tanh((log(a) - log(a_t))/Delta));

  // Simplified rho_mem evolution (full in ODE solver)
  // This is placeholder - full integration in perturbations
  *rho_mem = pba->rho_crit_today * pow(a, -3*(1+*w_mem));
  
  return _SUCCESS_;
}
