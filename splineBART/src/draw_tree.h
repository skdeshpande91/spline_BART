//
//  draw_tree.h
//

#ifndef tree_prior_h
#define tree_prior_h

#include <RcppArmadillo.h>
#include "rng.h"
#include "info.h"
#include "tree.h"
#include "funs.h"

void draw_tree(tree &x, xinfo &xi, double alpha, double beta, size_t D, RNG &gen);

#endif /* draw_tree_h */
