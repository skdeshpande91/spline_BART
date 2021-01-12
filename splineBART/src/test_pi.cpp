//
//  test_pi.cpp
//    Just checking that that including the #define _USE_MATH_DEFINES works
//    
//  Created by Sameer Deshpande on 1/5/21.
//

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <vector>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <cmath.h>

// [[Rcpp::export]]
double test_pi(size_t n = 0){
  double pi = M_PI;
  Rcpp::Rcout << "pi = " << pi << std::endl;
  return(log(2 * pi));
}
