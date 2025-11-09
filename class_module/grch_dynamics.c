/* File: grch_dynamics.c - GRCH cosmological perturbation module for CLASS */

#include "class.h" 
// Include other necessary headers (e.g., from your final CLASS implementation)

// GRCH parameters structure
struct grch_parameters {
    double at, delta, tau, S0;
};

// 1. Equation of State w(a)
double grch_w_of_a(double a, struct grch_parameters *pgrch) {
    // Implements w(a) = -0.5 * (1 + tanh((ln(a) - ln(at)) / delta))
    return -0.5 * (1.0 + tanh((log(a) - log(pgrch->at)) / pgrch->delta));
}

// 2. Clustering Parameter S(a)
double grch_S_of_a(double a, struct grch_parameters *pgrch) {
    // Implements the S(a) function from the paper:
    return pgrch->S0 * (1.0 - exp(-pow(a/pgrch->at, 2)));
}

// (Full C code for the evolution of rho_mem and the perturbation equations must go here)
