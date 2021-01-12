//
//  update_tree.h
// 1 January 2021: currently only supports grow/prune moves.
// eventually add support for other moves

#ifndef _GUARD_update_tree_h
#define _GUARD_update_tree_h

#include <RcppArmadillo.h>
#include "rng.h"
#include "info.h"
#include "tree.h"
#include "funs.h"
#include "mu_posterior.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <stdio.h>

void update_tree(tree &x, size_t &accept, double &sigma, std::vector<double> &theta, std::vector<size_t> &var_counts, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool debug);

void grow_tree(tree &x, size_t &accept, double &sigma, std::vector<double> &theta, std::vector<size_t> &var_counts, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool debug);

void prune_tree(tree &x, size_t &accept, double &sigma, std::vector<double> &theta, std::vector<size_t> &var_counts, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool debug);




#endif /* update_tree_h */
