//
//  test_backfit.cpp
//  
//
//  Created by Sameer Deshpande on 1/8/21.
//

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <vector>
#include <stdio.h>
#include "info.h"
#include "rng.h"
#include "funs.h"
#include "update_tree.h"
#include "update_sigma.h"

// [[Rcpp::export]]
Rcpp::List test_backfit(Rcpp::List Y_train,
                        Rcpp::List Phi_train,
                        arma::mat Z_train,
                        Rcpp::List cutpoints,
                        double alpha, double beta,
                        arma::mat K, double log_det_K, size_t K_ord, double tau, // tau is scaled by sqrt(m) internally
                        double nu_sigma, double lambda_sigma,
                        size_t M, bool debug)
{
  Rcpp::RNGScope scope;
  RNG gen;
  
  
  size_t n_train = Z_train.n_rows;
  size_t p = Z_train.n_cols;
  size_t D = K.n_rows;
  
  // create vectors that track Phi, residual, and products thereof
  std::vector<arma::mat> phi_train(n_train);
  std::vector<arma::vec> rf(n_train);
  std::vector<arma::mat> tphi_phi(n_train);
  std::vector<arma::vec> tphi_rf(n_train);
  
  // create pointer holding all entries in Z
  double* z_train_ptr = new double[n_train * p];
  
  size_t N = 0; // total number of observations
  
  for(size_t i = 0; i < n_train; i++){
    Rcpp::NumericMatrix tmp_phi = Phi_train[i];
    arma::mat tmp_phi2(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    
    Rcpp::NumericVector tmp_y = Y_train[i];
    arma::vec tmp_y2(tmp_y.begin(), tmp_y.size(), false, true);
    
    phi_train[i] = tmp_phi2;
    rf[i] = tmp_y2;
    
    // compute t(Phi) * y and t(Phi) * Phi only once and do it here to simplify input
    
    tphi_rf[i] = tmp_phi2.t() * tmp_y2;
    tphi_phi[i] = tmp_phi2.t() * tmp_phi2;
    
    N += rf[i].size();
    
    for(size_t j = 0; j < p; j++) z_train_ptr[j + p*i] = Z_train(i,j);
  }
  Rcpp::Rcout << "  Total of " << N << " observations from " << n_train << " subjects." << std::endl;
  
  
  // Read in and format the cutpoints
  xinfo xi;
  xi.resize(p);
  for(size_t j = 0; j < p; j++){
    Rcpp::NumericVector tmp = cutpoints[j];
    std::vector<double> tmp2;
    for(int jj = 0; jj < tmp.size(); jj++) tmp2.push_back(tmp[jj]);
    xi[j] = tmp2;
  }
  
  // create data object
  data_info di;
  di.n = n_train;
  di.N = N;
  di.p = p;
  di.D = D;
  di.z = &z_train_ptr[0];
  di.phi = &phi_train;
  di.rf = &rf;
  di.tphi_phi = &tphi_phi;
  di.tphi_rf = &tphi_rf;
  
  // create tree_prior object
  tree_prior_info tree_pi;
  tree_pi.alpha = alpha;
  tree_pi.beta = beta;
  tree_pi.K = K;
  tree_pi.log_det_K = log_det_K;
  tree_pi.tau = tau/sqrt( (double) M);
  tree_pi.K_ord = K_ord;
  
  // splitting variable probabilities and variable counts
  std::vector<double> theta(p, 1.0/( (double) p));
  std::vector<size_t> var_counts(p, 0);
  
  // create vector of trees
  std::vector<tree> t_vec(M);
  tree::npv bnv;
  for(size_t m = 0; m < M; m++){
    t_vec[m].getbots(bnv);
    for(size_t l = 0; l < bnv.size(); l++){
      bnv[l]->setm(arma::zeros<arma::vec>(D)); // initialize initial fit of each tree to be 0
    }
  }
  
  double sigma = 1.0;
  sigma_prior_info sigma_pi;
  sigma_pi.nu = nu_sigma;
  sigma_pi.lambda = lambda_sigma;
  
  
  // create objects to hold (i) mu, the fit of a single tree; (ii) phi_mu, Phi_i * g(z;T,mu), (iii) t(Phi_i) * Phi_i * m
  
  arma::mat mu = arma::zeros<arma::mat>(D, n_train); // holds fit a single tree
  std::vector<arma::vec> phi_mu(n_train); // holds Phi_i * mu. This gets added and then subtracted from rf
  arma::mat tphi_phi_mu(D, n_train); // this gets added and then subtract from tphi_rf
  
  // some objects useful for convergence related stuff
  size_t accept = 0;
  
  
  for(size_t m = 0; m < M; m++){
    // get fit of m-th tree
    get_fit(mu, phi_mu, tphi_phi_mu, t_vec[m], xi, di);
    // remove fit of m-th tree from residual
    for(size_t i = 0; i < n_train; i++){
      rf[i] += phi_mu[i];
      tphi_rf[i] += tphi_phi_mu.col(i);
    }
    Rcpp::Rcout << "updating tree m = " << m;
    
    // at this point rf[i] is y_i - Phi_i * (sum of all but m-th tree)
    // at this point tphi_rf[i] is Phi_i' * (y_i - phi_i * (sum of all but m-th tree))
    update_tree(t_vec[m], accept, sigma, theta, var_counts, xi, di, tree_pi, gen, debug);
    Rcpp::Rcout << "  accept = " << accept << std::endl;
    
    get_fit(mu, phi_mu, tphi_phi_mu, t_vec[m], xi, di);
    // add the fit of m-th tree back to residual
    for(size_t i = 0; i < n_train; i++){
      rf[i] -= phi_mu[i];
      tphi_rf[i] -= tphi_phi_mu.col(i);
    }
    // at this point rf[i] = y_i - Phi_i * (sum of all trees)
    // at this point tphi_rf[i] = Phi_i' * (y_i - Phi_i * (sum of all trees))
  }
  
  // update sigma
  update_sigma(sigma, sigma_pi, di, gen);
  
  
  // let us check that we actually have computed the full residuals correctly
  Rcpp::List rf_out(n_train);
  
  for(size_t i = 0; i < n_train; i++){
    rf_out[i] = rf[i];
  }
  
  arma::mat beta_out = arma::zeros<arma::mat>(D, n_train);
  for(size_t m = 0; m < M; m++){
    get_beta_fit(mu, t_vec[m], xi, di);
    beta_out += mu; // add fit of m-th tree to output
  }
  
  Rcpp::List results;
  results["beta"] = beta_out;
  results["sigma"] = sigma;
  results["rf"] = rf_out; // continuously updated during tree updates
  return(results);
  
}
