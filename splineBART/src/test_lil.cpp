//
//  test_lil.cpp
//  
//
//  Created by Sameer Deshpande on 1/5/21.
//

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <vector>
#include <stdio.h>
#include "rng.h"
#include "info.h"
#include "tree.h"
#include "funs.h"
#include "mu_posterior.h"
#include "draw_tree.h"

// [[Rcpp::export]]
Rcpp::List test_lil(Rcpp::List Phi_train,
                             arma::mat Z_train,
                             Rcpp::List cutpoints,
                             size_t D,
                             arma::mat K,
                             double log_det_K,
                             size_t K_ord,
                             double tau,
                             Rcpp::List tPhi_rf,
                             double sigma)
{
  Rcpp::RNGScope scope;
  RNG gen;
  
  
  size_t n_train = Z_train.n_rows;
  size_t p = Z_train.n_cols;
  
  
  // Prepare vectors holding Phi, Phi' Phi, and Phi' rf
  // prepare pointer containing Z information
  
  std::vector<arma::mat> phi_train(n_train);
  std::vector<arma::mat> tphi_phi(n_train);
  std::vector<arma::vec> tphi_rf(n_train);
  
  double* z_train_ptr = new double[n_train * p];
  
  for(size_t i = 0; i < n_train; i++){
    Rcpp::NumericMatrix tmp_phi = Phi_train[i];
    arma::mat tmp_phi2(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    
    phi_train[i] = tmp_phi2;
    // compute t(Phi) * Phi only once and do it here to simplify input
    tphi_phi[i] = tmp_phi2.t() * tmp_phi2;
    
    Rcpp::NumericVector tmp_tphi_rf = tPhi_rf[i];
    arma::vec tmp_tphi_rf2(tmp_tphi_rf.begin(), tmp_tphi_rf.size(), false, true);
    tphi_rf[i] = tmp_tphi_rf2;
    
    for(size_t j = 0; j < p; j++) z_train_ptr[j + p*i] = Z_train(i,j);
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
  
  // create data object
  data_info di;
  di.n = n_train;
  di.p = p;
  di.D = D;
  di.z = &z_train_ptr[0];
  di.phi = &phi_train;
  di.tphi_phi = &tphi_phi;
  di.tphi_rf = &tphi_rf;
  
  // create tree prior object
  tree_prior_info tree_pi;
  tree_pi.K = K;
  tree_pi.log_det_K = log_det_K;
  tree_pi.tau = tau;
  tree_pi.K_ord = K_ord;
  
  tree t;
  tree::npv bnv;
  std::vector<sinfo> sv;
  bool empty_leaf = true;
  size_t counter = 0;
  size_t L = 0;
  // for testing, when we randomly generate tree
  // we need to ensure all leafs have at least one element
  while(empty_leaf && (counter < 5000)){
    empty_leaf = false;
    draw_tree(t, xi, tree_pi.alpha, tree_pi.beta, D, gen);
    allsuff(t, xi, di, bnv, sv);
    L = bnv.size();
    for(size_t l = 0; l < L; l++){
      if(sv[l].n == 0){
        empty_leaf = true;
        break;
      }
    }
    counter++;
  }
  if(!empty_leaf){
    // at this point our tree does not have an empty leaf node
    Rcpp::Rcout << "Took " << counter << " tries to get tree with non-empty leaves" << std::endl;
    t.print(true);
    Rcpp::NumericVector lil_vec(L);
    Rcpp::NumericVector leaf_ix(n_train);
    for(size_t l = 0; l < L; l++){
      lil_vec[l] = compute_lil(sv[l], sigma, di, tree_pi);
      for(size_t ix = 0; ix < sv[l].n; ix++) leaf_ix[sv[l].I[ix]] = l+1; // R is 1-indexed!
    }
    // this loops over leaf in left-to-right order
    // for testing computations in R, we need to know which observations
    // go to which leaf. we will use leaf_ix
    Rcpp::List results;
    results["lil"] = lil_vec;
    results["leaf_ix"] = leaf_ix;
    return(results);
  } else{
    Rcpp::Rcout << " counter = " << counter << std::endl;
    Rcpp::stop("after 5000 tries, tree has empty leaves");
  }

}



