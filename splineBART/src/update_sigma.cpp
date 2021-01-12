//
//  update_sigma.cpp
//  
//
//  Created by Sameer Deshpande on 1/7/21.
//

#include "update_sigma.h"
void update_sigma(double &sigma, sigma_prior_info &sigma_pi, data_info &di, RNG &gen)
{
  double scale_post = sigma_pi.lambda * sigma_pi.nu;
  double nu_post = sigma_pi.nu + di.N;
  
  for(size_t i = 0; i < di.n; i++){
    scale_post += arma::as_scalar(arma::dot(di.rf->at(i), di.rf->at(i)));
  }
  sigma = sqrt( (scale_post)/gen.chi_square(nu_post));
}
