//
//  mu_posterior.h
//  
//

#ifndef GUARD_mu_posterior_h
#define GUARD_mu_posterior_h

#include<RcppArmadillo.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include "tree.h"
#include "info.h"
#include <stdio.h>



void get_post_params(arma::vec &m, arma::mat &P, sinfo &si, double &sigma, data_info &di, tree_prior_info &tree_pi);


// rewrite compute_lil in a way that wraps get_post_params.
double compute_lil(sinfo& si, double &sigma, data_info &di, tree_prior_info &tree_pi);


#endif
