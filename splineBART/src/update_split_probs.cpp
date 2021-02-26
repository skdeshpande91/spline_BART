//
//  update_split_probs.cpp
//    Update the split probabilities, theta
//  Created by Sameer Deshpande on 4/7/20.


#include "update_split_probs.h"


void update_eta(double &eta, double &rho_eta, std::vector<double> &theta, size_t &p, size_t &N_u, double &a, double &b, RNG &gen)
{
  double sum_log_theta = 0.0;
  for(size_t j = 0; j < p; j++) sum_log_theta += log(theta[j]);
  
  std::vector<double> eta_log_prob(N_u - 1);
  std::vector<double> eta_prob(N_u - 1);
  
  double max_log_prob = 0.0;
  
  double tmp_u = 0.0;
  double tmp_eta = 0.0;
  double tmp_sum = 0.0;
  
  for(size_t u_ix = 0; u_ix < N_u - 1; u_ix++){
    tmp_u = ( (double) u_ix + 1.0)/( (double) N_u);
    tmp_eta = rho_eta * tmp_u/(1.0 - tmp_u);
    
    eta_log_prob[u_ix] = lgamma(tmp_eta) - ( (double) p) * lgamma(tmp_eta/( (double) p));
    eta_log_prob[u_ix] += tmp_eta/( (double) p) * sum_log_theta;
    eta_log_prob[u_ix] += (a - 1.0) * log(tmp_u) + (b + 1) * log(1.0 - tmp_u);
    
    if( (u_ix == 0) || (max_log_prob < eta_log_prob[u_ix]) ) max_log_prob = eta_log_prob[u_ix];
  }
  
  for(size_t u_ix = 0; u_ix < N_u - 1; u_ix++){
    eta_prob[u_ix] = exp(eta_log_prob[u_ix] - max_log_prob);
    tmp_sum += eta_prob[u_ix];
  }
  
  for(size_t u_ix = 0; u_ix < N_u - 1; u_ix++) eta_prob[u_ix] /= tmp_sum;
  
  size_t num_etas = N_u - 1;
  size_t u_ix_new = gen.multinom(num_etas, eta_prob);
  eta = rho_eta * ( (double) u_ix_new + 1.0)/( (double) N_u);
  
}



void update_theta(std::vector<double> &theta, std::vector<size_t> &var_counts, double &eta, size_t &p, RNG &gen)
{
  std::vector<double> eta_new(p, eta/ ( (double) p));// this is basically the prior value
  for(size_t j = 0; j < p; j++) eta_new[j] += (double) var_counts[j];
  gen.dirichlet(theta, eta_new, p);
}
