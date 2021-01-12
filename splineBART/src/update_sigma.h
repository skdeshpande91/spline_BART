#ifndef GUARD_update_sigma_h
#define GUARD_update_sigma_h

#include <RcppArmadillo.h>
#include "rng.h"
#include "info.h"

#include <stdio.h>

void update_sigma(double &sigma, sigma_prior_info &sigma_pi, data_info &di, RNG &gen);


#endif /* GUARD_update_sigma_h */
