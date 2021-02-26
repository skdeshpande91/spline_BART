//
//  mu_posterior.cpp
//  
//
//  Created by Sameer Deshpande on 2/3/20.
//

#include "mu_posterior.h"


// function only returns m and P.
// we will compute log-likelihood based on these two values
// we will also need to draw multivariate normal samples based on these two values

void get_post_params(arma::vec &m, arma::mat &P, sinfo &si,  double &sigma, data_info &di, tree_prior_info &tree_pi)
{
  // re-set mean and prec_chol just in case we don't do this out of the loop
  
  P = ((double) di.M * 1.0/pow(tree_pi.tau, 2.0)) * tree_pi.K; // prior precision matrix of the spline coefficients.
  m = arma::zeros<arma::vec>(di.D);
  
  for(size_t ix = 0; ix < si.n; ix++){
    P += 1.0/pow(sigma, 2.0) * di.tphi_phi->at(si.I[ix]); // adds sigma^-2 * Phi_i'Phi_i to running precision matrix
    m += 1.0/pow(sigma, 2.0) * di.tphi_rf->at(si.I[ix]); // adds sigma^-2 * Phi_i'r_i to running sum m
  }
  
  // posterior of mu is N(P^-1m, P^-1)
  
}

double compute_lil(sinfo& si, double &sigma, data_info &di,  tree_prior_info &tree_pi){
  
  arma::vec m = arma::zeros<arma::vec>(di.D);
  arma::mat P = arma::zeros<arma::mat>(di.D, di.D);
  
  get_post_params(m, P, si, sigma, di, tree_pi);
  
  arma::mat L = arma::chol(P, "lower"); // P = L L'
  arma::vec nu = arma::solve(arma::trimatl(L), m); // nu = L^-1 m.
  double quad_form = arma::as_scalar(arma::dot(nu,nu)); // nu'nu = m' (L^-1)'(L^-1)m = m' P^-1 m
  
  // lil has a factor of -0.5 * log |P| = -1.0 * log |L|
  // since L is lower triangular this is just -1.0 * arma::accu(arma::log(L.diag())
  double log_det_L = arma::accu(arma::log(L.diag())); // take advantage of the fact L is lower triangular
  
  // now deal with the extra factors
  double lil = -0.5 * ( (double) di.D - (double) tree_pi.rank_K) * log(2.0 * M_PI);
  lil += 0.5 * tree_pi.log_det_K;
  lil += ( (double) tree_pi.rank_K ) * (0.5 * log( (double) di.M) - log(tree_pi.tau));
  lil += 0.5 * quad_form - 1.0 * log_det_L;
  return(lil);
  
}




