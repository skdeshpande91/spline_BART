#include <cmath>
#include "rng.h"

double RNG::uniform(double x, double y ){
  return R::runif(x,y);
}
double RNG::normal(double mu, double sd ){
  return R::rnorm(mu,sd);
}

double RNG::gamma(double shape, double scale)
{ return R::rgamma(shape, 1)*scale; }

double RNG::beta(double a1, double a2){
  const double x1 = gamma(a1, 1); return (x1 / (x1 + gamma(a2, 1)));
}

double RNG::chi_square(double df){
  return R::rchisq(df);
}

void RNG::dirichlet(std::vector<double> &theta, const std::vector<double> &alpha, const size_t &R)
{
  // quick check of size
  if(alpha.size() != R){
    Rcpp::Rcout << "[RNG::dirichlet]: alpha.size = " << alpha.size() << " R = " << R << std::endl;
    Rcpp::stop("alpha must be of length R!");
  }
  
  theta.clear();
  theta.resize(R);
  std::vector<double> tmp_gamma(R);
  double tmp_sum = 0.0;
  for(size_t r = 0; r < R; r++){
    tmp_gamma[r] = gamma(alpha[r], 1.0);
    tmp_sum += tmp_gamma[r];
  }
  for(size_t r = 0; r < R; r++) theta[r] = tmp_gamma[r]/tmp_sum;
}

size_t RNG::multinom(const size_t &R, const std::vector<double> &probs){
  size_t x = 0;
  double cumsum = 0.0;
  double unif = uniform(0,1);
  for(size_t r = 0; r < R; r++){
    cumsum += probs[r];
    if(unif < cumsum){
      x = r;
      break;
    }
  }
  return(x);
}

arma::vec RNG::std_norm_vec(size_t d){
  arma::vec results(d);
  for(size_t i = 0; i < d; i++) results(i) = normal(0.0,1.0);
  return(results);
}

arma::vec RNG::mvnormal(arma::vec m, arma::mat P){
  size_t d = m.size();
  if( (P.n_rows != d) | (P.n_cols != d)){
    Rcpp::Rcout << "m.size() = " << d << " P.nrow = " << P.n_rows << " P.n_cols = " << P.n_cols << std::endl;
    Rcpp::stop("[RNG::mvnormal]: m & P have incompatible dimensions!");
  }
  
  arma::mat L = arma::chol(P, "lower"); // P = L L.t()
  arma::vec nu = arma::solve(arma::trimatl(L), m); // nu = L^-1 m.
  arma::vec mu = arma::solve(arma::trimatu(L.t()), nu); // mu = (L')^-1 L^-1 m = P^-1 m
  arma::vec z = arma::solve(arma::trimatu(L.t()), std_norm_vec(d)); // Cov(z) = (L')^-1 ((L')^-1)' = (L')^-1 L^-1 = P^-1
  
  arma::vec results = mu + z;
  return(results);
}

