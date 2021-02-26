
#ifndef _GUARD_update_split_probs_h
#define _GUARD_update_split_probs_h
#include "rng.h"
#include <stdio.h>


void update_eta(double &eta, double &rho_eta, std::vector<double> &theta, size_t &p, size_t &N_u, double &a, double &b, RNG &gen);
void update_theta(std::vector<double> &theta, std::vector<size_t> &var_counts, double &eta, size_t &R, RNG &gen);



#endif /* update_split_probs_h */
