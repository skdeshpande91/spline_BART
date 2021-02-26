//
//  spline_bart_fixed.cpp
//     Fixed split probabilities

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <vector>
#include <stdio.h>
#include "update_tree.h"
#include "update_sigma.h"
// these two includes should capture everything else

// [[Rcpp::export(name = ".fixed_theta_adapt_tau")]]
Rcpp::List fixed_theta_adapt_tau(Rcpp::List Y_train,
                                 Rcpp::List Phi_train,
                                 arma::mat Z_train,
                                 arma::mat Z_test,
                                 Rcpp::List cutpoints,
                                 double alpha, double beta,
                                 arma::mat K, double log_det_K, size_t rank_K,
                                 double nu_tau, double lambda_tau,
                                 double nu_sigma, double lambda_sigma,
                                 size_t M, size_t nd, size_t burn,
                                 bool verbose, size_t print_every, bool debug = false)
{
  Rcpp::RNGScope scope;
  RNG gen;
  
  size_t n_train = Z_train.n_rows;
  size_t n_test = Z_test.n_rows;
  size_t p = Z_train.n_cols;
  size_t D = K.n_rows;
    
  // create vectors that track Phi, residual, and products thereof
  std::vector<arma::mat> phi(n_train);
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
    
    phi[i] = tmp_phi2;
    rf[i] = tmp_y2;
    
    // compute t(Phi) * y and t(Phi) * Phi only once and do it here to simplify input
    
    tphi_rf[i] = tmp_phi2.t() * tmp_y2;
    tphi_phi[i] = tmp_phi2.t() * tmp_phi2;
    
    N += rf[i].size();
    
    for(size_t j = 0; j < p; j++) z_train_ptr[j + p*i] = Z_train(i,j);
  }
  if(verbose) Rcpp::Rcout << "  Total of " << N << " training observations from " << n_train << " subjects." << std::endl;
  
  // prepare pointer for Z_test
  double* z_test_ptr = new double[n_test * p];
  for(size_t i = 0; i < n_test; i++){
    for(size_t j = 0; j < p; j++) z_test_ptr[j + p * i] = Z_test(i,j);
  }
  
  // Read in and format the cutpoints
  xinfo xi;
  xi.resize(p);
  for(size_t j = 0; j < p; j++){
    Rcpp::NumericVector tmp = cutpoints[j];
    std::vector<double> tmp2;
    for(int jj = 0; jj < tmp.size(); jj++) tmp2.push_back(tmp[jj]);
    xi[j] = tmp2;
  }
  
  // create data object for training data
  data_info di;
  di.n = n_train;
  di.N = N;
  di.p = p;
  di.D = D;
  di.M = M;
  di.z = &z_train_ptr[0];
  di.phi = &phi;
  di.rf = &rf;
  di.tphi_phi = &tphi_phi;
  di.tphi_rf = &tphi_rf;
  
  // create data object for testing data
  data_info dip;
  dip.n = n_test;
  dip.p = p;
  dip.D = D;
  dip.z = &z_test_ptr[0];
  
  // create tree_prior object
  tree_prior_info tree_pi;
  tree_pi.alpha = alpha;
  tree_pi.beta = beta;
  tree_pi.K = K;
  tree_pi.log_det_K = log_det_K;
  tree_pi.rank_K = rank_K;
  
  tree_pi.tau = 1.0;
  tree_pi.nu_tau = nu_tau;
  tree_pi.lambda_tau = lambda_tau;
  tree_pi.nu_post = nu_tau;
  tree_pi.scale_post = nu_tau * lambda_tau;
  
  
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
  
  // residual standard deviation
  double sigma = 1.0;
  sigma_prior_info sigma_pi;
  sigma_pi.nu = nu_sigma;
  sigma_pi.lambda = lambda_sigma;
  
  // create objects to hold (i) mu, fit of a single tree, (ii) phi_mi, Phi_i * g(z;T, mu), and (iii) t(Phi_i) * Phi_i * m
  
  arma::mat mu_train = arma::zeros<arma::mat>(D, n_train); // holds fit of a single tree for training
  arma::mat mu_test = arma::zeros<arma::mat>(D, n_test);
  
  std::vector<arma::vec> phi_mu(n_train); // don't need test set version of this
  arma::mat tphi_phi_mu(D, n_train); // don't need test set version of this
  
  size_t accept = 0;
  
  // create output containers
  arma::cube beta_train_samples(D, n_train, nd);
  arma::cube beta_test_samples(D, n_test, nd);
  
  Rcpp::NumericVector sigma_samples(nd + burn);
  Rcpp::NumericVector tau_samples(nd + burn);
  
  if(verbose) Rcpp::Rcout << "[splineBART]: Starting MCMC" << std::endl;
  for(size_t iter = 0; iter < nd + burn; iter++){
    if(verbose){
      Rcpp::checkUserInterrupt();
      if( (iter < burn) && (iter%print_every == 0)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Burn-in" << std::endl;
      else if( ( (iter > burn) && (iter%print_every == 0) ) || (iter == burn)) Rcpp::Rcout << "  MCMC Iteration: " << iter << " of " << nd + burn << "; Sampling" << std::endl;
    }
    
    // prior to updating each tree, we need to reset the running df and scale of the posterior on tau^2
    tree_pi.nu_post = tree_pi.nu_tau;
    tree_pi.scale_post = tree_pi.nu_tau * tree_pi.lambda_tau;
    for(size_t m = 0; m < M; m++){
      get_fit(phi_mu, tphi_phi_mu, t_vec[m], xi, di); // get fit of m-th tree
      
      // temporarily remove fit of m-th tree from residuals
      for(size_t i = 0; i < n_train; i++){
        rf[i] += phi_mu[i];
        tphi_rf[i] += tphi_phi_mu.col(i);
      }
      
      // at this point:
      // rf[i] is y_i - Phi_i * (sum of all but m-th tree)
      // tphi_rf[i] is Phi_i' * (y_i - phi_i * (sum of all but m-th tree))
      update_tree(t_vec[m], accept, sigma, theta, var_counts, xi, di, tree_pi, gen, debug);

      // now the tree has been updated, we need to get fit and fix the residuals
      get_fit(phi_mu, tphi_phi_mu, t_vec[m], xi, di);
      for(size_t i = 0; i < n_train; i++){
        rf[i] -= phi_mu[i];
        tphi_rf[i] -= tphi_phi_mu.col(i);
      }
      // at this point
      // rf[i] = y_i - Phi_i * (sum of all trees)
      // tphi_rf[i] = Phi_i' * (y_i - Phi_i * (sum of all trees))
      
    } // closes loop over trees
    
    // now that all trees have been updated, we draw a new value of tau
    tree_pi.tau = sqrt( (tree_pi.scale_post)/gen.chi_square(tree_pi.nu_post));
    
    // now that all trees have been updated, we can update sigma
    update_sigma(sigma, sigma_pi, di, gen);
    
    sigma_samples(iter) = sigma; // save the value of sigma
    tau_samples(iter) = tree_pi.tau; // save the value of tau
    
    if(iter >= burn){
      beta_train_samples.slice(iter-burn).zeros(D, n_train);
      beta_test_samples.slice(iter-burn).zeros(D, n_test);
      for(size_t m = 0; m < M; m++){
        get_beta_fit(mu_train, t_vec[m], xi, di);
        get_beta_fit(mu_test, t_vec[m], xi, dip);
        
        beta_train_samples.slice(iter-burn) += mu_train;
        beta_test_samples.slice(iter-burn) += mu_test;
      }
    } // closes if checking that we're saving samples of beta
  } // closes main MCMC loop
  if(verbose) Rcpp::Rcout << "[splineBART]: Finished MCMC!" << std::endl;
  
  
  Rcpp::List results;
  results["beta_train_samples"] = beta_train_samples;
  results["beta_test_samples"] = beta_test_samples;
  results["sigma_samples"] = sigma_samples;
  results["tau_samples"] = tau_samples;
  return(results);
}
