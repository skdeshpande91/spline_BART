//
//  summarize_posterior.cpp
//  
//
//  Created by Sameer Deshpande on 2/25/21.
//
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <vector>
#include <stdio.h>
#include "rng.h"

// [[Rcpp::export(name = ".summarize_fit")]]
Rcpp::List summarize_fit(Rcpp::List beta_list,
                         Rcpp::List Phi_list,
                         arma::vec probs,
                         size_t n_samples,
                         double y_mean,
                         double y_sd)
{
  size_t n = Phi_list.size();
  size_t n_chains = beta_list.size();
  size_t ni = 0;
  
  size_t n_quant = probs.size();
  
  Rcpp::List fit_summary(n);
  for(size_t i = 0; i < n; i++){
    // read in the Phi matrix
    Rcpp::NumericMatrix tmp_phi = Phi_list[i];
    arma::mat phi(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    ni = phi.n_rows;
    
    arma::mat fit = arma::zeros<arma::mat>(ni, n_chains * n_samples);
    
    for(size_t c = 0; c < n_chains; c++){
      Rcpp::NumericVector tmp_beta = beta_list[c];
      Rcpp::IntegerVector tmp_dim = tmp_beta.attr("dim");
      arma::cube beta(tmp_beta.begin(), tmp_dim[0], tmp_dim[1], tmp_dim[2], false, true);
      arma::mat beta_i = beta(arma::span::all, arma::span(i), arma::span::all); // D x n_samples
      fit.cols(c*n_samples, (c+1)*n_samples-1) = y_mean + y_sd * phi * beta_i;
    }
    
    arma::mat tmp_sum = arma::zeros<arma::mat>(ni, n_quant + 1);
    
    tmp_sum.col(0) = arma::mean(fit,1);
    tmp_sum.cols(1,n_quant) = arma::quantile(fit, probs, 1);
    fit_summary(i) = tmp_sum;
  }
  return(fit_summary);
}

// [[Rcpp::export(name = ".summarize_fit_ystar")]]
Rcpp::List summarize_fit_ystar(Rcpp::List beta_list,
                               Rcpp::List Phi_list,
                               Rcpp::List sigma_list,
                               arma::vec probs,
                               size_t n_samples,
                               double y_mean,
                               double y_sd)
{
  Rcpp::RNGScope scope;
  RNG gen;

  size_t n = Phi_list.size();
  size_t n_chains = beta_list.size();
  size_t ni = 0;
  
  size_t n_quant = probs.size();
  
  Rcpp::List fit_summary(n);
  Rcpp::List ystar_summary(n);
  
  
  for(size_t i = 0; i < n; i++){
    // read in the Phi matrix
    Rcpp::NumericMatrix tmp_phi = Phi_list[i];
    arma::mat phi(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    ni = phi.n_rows;
    
    arma::mat fit = arma::zeros<arma::mat>(ni, n_chains * n_samples);
    arma::mat ystar = arma::zeros<arma::mat>(ni, n_chains * n_samples);
    
    for(size_t c = 0; c < n_chains; c++){
      Rcpp::NumericVector tmp_sigma = sigma_list[c];
      arma::vec sigma(tmp_sigma.begin(), n_samples, false, true);
      
      Rcpp::NumericVector tmp_beta = beta_list[c];
      Rcpp::IntegerVector tmp_dim = tmp_beta.attr("dim");
      arma::cube beta(tmp_beta.begin(), tmp_dim[0], tmp_dim[1], tmp_dim[2], false, true);
      arma::mat beta_i = beta(arma::span::all, arma::span(i), arma::span::all); // D x n_samples
      
      arma::mat tmp_fit  = phi * beta_i;
      fit.cols(c*n_samples, (c+1)*n_samples-1) = y_mean + y_sd * tmp_fit;
      
      arma::mat eps = gen.std_norm_mat(ni, n_samples);
      eps.each_row() %= sigma.t();
      
      ystar.cols(c*n_samples, (c+1)*n_samples-1) = y_mean + y_sd * tmp_fit + y_sd * eps;
      
      //for(size_t j = 0; j < n_samples; j++) ystar.col(c*n_samples + j) = sigma[j] * gen.std_norm_vec(ni);
      
    }
    
    arma::mat tmp_sum = arma::zeros<arma::mat>(ni, n_quant + 1);
    
    tmp_sum.col(0) = arma::mean(fit,1);
    tmp_sum.cols(1,n_quant) = arma::quantile(fit, probs, 1);
    fit_summary[i] = tmp_sum;
    
    tmp_sum.col(0) = arma::mean(ystar,1);
    tmp_sum.cols(1,n_quant) = arma::quantile(ystar, probs, 1);
    ystar_summary[i] = tmp_sum;

  }
  Rcpp::List results;
  results["fit"] = fit_summary;
  results["ystar"] = ystar_summary;
  return(results);
  
}

