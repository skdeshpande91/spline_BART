#ifndef GUARD_info_h
#define GUARD_info_h

#include <RcppArmadillo.h>

//============================================================
//data
//============================================================


class data_info{
public:
  size_t n; // number of surfaces
  size_t N; // total number of observations
  size_t p; // number of predictors
  size_t D; // number of basis vectors

  double *z; // z[j + i*p] is Z_ij
  std::vector<arma::mat>* phi; // points to vector of the Phi matrices
  std::vector<arma::vec>* rf;// points to the full residual vector.
  std::vector<arma::mat>* tphi_phi; // points to vector of t(Phi) * Phi
  std::vector<arma::vec>* tphi_rf; // points to vector of t(Phi) * (residual). constantly updating in main loop

  data_info(){n = 0; p = 0; D = 0; z = 0; phi = 0; rf = 0; tphi_phi = 0; tphi_rf = 0;}
};

class tree_prior_info{
public:
  double pbd; // probability of birth/death move
  double pb; // probability of birth given birth/death move
  double alpha;
  double beta;
  arma::mat K; // precision matrix for the mu vector in each leaf
  double log_det_K; // holds log of generalized determinant of Omega. will be passed in by user
  double tau; // variance scaling (if we ever use it)
  size_t K_ord; // order of K (typically always equal to 1)
 
  // t(Phi_i)*r_i is needed to compute the posterior mean of mu
  tree_prior_info(){pbd = 1.0; pb = 0.5; alpha = 0.95; beta = 2.0; K = arma::zeros<arma::mat>(1,1); log_det_K = 0.0; tau = 1.0; K_ord = 1;}
};

class sigma_prior_info{
public:
  double nu;
  double lambda;
  sigma_prior_info(){nu = 3.0; lambda = 1.0;}
};

class sinfo{
public:
  size_t n; // how many surfaces land in this node
  std::vector<size_t> I; // holds the indices of the surfaces that land in this node
  sinfo(){n=0;I = std::vector<size_t>(1);}
};

#endif






