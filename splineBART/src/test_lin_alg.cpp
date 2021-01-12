//
//  test_lin_alg.cpp
//    Test that our linear algebra is implemented correctly
//    Given a matrix P and vector m, we need to compute things like
//        1. m' P^-1 m (quad_form)
//        2. P^-1 m (post_mean)
//        3. log det P

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
Rcpp::List test_lin_alg(arma::mat P,
                        arma::vec m)
{
  arma::mat P_inv = arma::inv_sympd(P);
  double quad_form1 = arma::as_scalar(m.t() * P_inv * m);
  arma::vec post_mean1 = P_inv * m;
  double log_det_P = 0.0;
  double tmp_sgn = 1.0;
  arma::log_det(log_det_P, tmp_sgn, P);
  
  
  arma::mat L = arma::chol(P, "lower");
  arma::vec nu = arma::solve(arma::trimatl(L), m); // nu = L^-1 m.
  arma::vec post_mean2 = arma::solve(arma::trimatu(L.t()), nu); // mu = (L')^-1 L^-1 m
  double quad_form2 = arma::as_scalar(arma::dot(nu,nu));
  
  double log_det_L1 = 0.0;
  arma::log_det(log_det_L1, tmp_sgn, L);
  double log_det_L2 = arma::accu(arma::log(L.diag()));
  
  Rcpp::List results;
  results["quad_form1"] = quad_form1;
  results["quad_form2"] = quad_form2;
  results["post_mean1"] = post_mean1;
  results["post_mean2"] = post_mean2;
  results["log_det_L1"] = log_det_L1;
  results["log_det_L2"] = log_det_L2;
  results["log_det_P"] = log_det_P;
  
  // we need to check that quad_form1 = quad_form2, post_mean1 = post_mean2
  // and log_det_L1 = log_detL2 = 0.5 * log_det_P
  
  return(results);
  
}


