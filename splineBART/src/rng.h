#ifndef RNG_H
#define RNG_H
#include <RcppArmadillo.h>

using std::vector;

class RNG
{
public:
  // continuous distributions
  double uniform(double x = 0.0, double y = 1.0);
  double normal(double mu = 0.0, double sd = 1.0);
  double gamma(double shape, double scale);
  double chi_square(double df);
  
  double beta(double a1, double a2);
  
  void dirichlet(std::vector<double> &theta, const std::vector<double> &alpha, const size_t &R);
  
  // discrete
  size_t multinom(const size_t &R, const std::vector<double> &probs);
  
  // vector of standard normals
  arma::vec std_norm_vec(size_t d);
  
  // sample from multivariate normal N(P^-1m, P^-1)
  arma::vec mvnormal(arma::vec m, arma::mat P);
  
  // whereas jestarling's code involved a multivariate normal samples
  // it will be easier to do that within the update_trees function itself
  // since we would have already had to compute (a) the posterior mean and (b) cholesky decomposition of precision matrix
 
};
#endif // RNG_H
