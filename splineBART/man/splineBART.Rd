\name{splineBART}
\alias{splineBART}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Fit a Bayesian additive treed spline model.
}
\description{
The Bayesian additive treed spline model extends BART to perform targetted smoothing. 
Each regression tree in the ensemble partitions a space of predictions Z and outputs a smooth function of a scalar variable X. 
The smooth functions are spline basis expansions.
}
\usage{
splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints,
           K, log_det_K, rank_K,
           M = 50, alpha = 0.95, beta = 2,
           nu_sigma = 3, sigma_hat = 1, sigquant = 0.9,
           nd = 1000, burn = 250, 
           verbose = TRUE, print_every = 50,
           fixed_theta = FALSE, fixed_tau = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y_train}{List containing vector of observed outcomes for each sample.}
  \item{Phi_train}{List containing spline basis design matrix for each sample.}
  \item{Z_train}{Matrix of predictors, with one row per training set sample.}
  \item{Z_test}{Matrix of predictors, with one row per testing set sample.}
  \item{cutpoints}{L containing all allowable cutpoints for decision trees.}
  \item{K}{Prior predicison matrix for vector of spline coefficients in leaves of each tree. May be singular (se e details).}
  \item{log_det_K}{Log of generalized determinant of K (see details).}
  \item{rank_K}{Rank of prior predicision matrix}
  \item{M}{Number of trees in ensemble. Default is 50.}
  \item{alpha}{First hyperparameter of branching process prior for decision trees. Default is 0.95}
  \item{beta}{Second hyperparameter of branching process prior for decision trees. Default is 2}
  \item{nu_sigma}{Prior deegrees of freedom for residual standard deviation. Default is 3.}
  \item{sigma_hat}{Overestimate of residual standard deviation. Default is 1, as outcomes are standardized internally.}
  \item{sigquant}{How much prior probability is placed on the event (sigma < sigma_hat). Default is 0.9}
  \item{nd}{Number of MCMC samples to save. Default is 1000.}
  \item{burn}{Number of MCMC iterations to discard as burn-in. Default is 500.}
  \item{nd}{Number of MCMC samples to save. Default is 1000.}
  \item{verbose}{Boolean indicating whether MCMC progress is to be printed. Default is TRUE}
  \item{print_every}{Number of MCMC iterations between progress statements are printed. Default is 50.}
  \item{fixed_theta}{Boolean indicating whether or not the split probability indices are to be fixed or learned from data. Default is FALSE. If TRUE, must specify the theta argument.}
  \item{fixed_tau}{Boolean indicating whether or not scale parameter tau in spline coefficient prior should be kept fixed or learned from data. Default is FALSE. If TRUE, must specify the tau argument.}
  \item{theta}{Vector split index probabilitiy. Must be a vector of length ncol(Z_train). Only used if fixed_theta = TRUE. Otherwise, default is NULL}
  \item{a}{First hyperparameter for discretized Beta prior on split probability concentration parameter. Default is 1.}
  \item{b}{Second hyperparameter for discretized Beta prior on split probability concentration. Default is p.}
  \item{N_u}{Number of grid points for split probability concentration parameter. Default is 100.}
  \item{rho_eta}{Additional hyperparmeter used to define split probability concentration parameter. Default is p.}
  \item{tau}{Prior scale parameter for spline coefficients. Must be specified if fixed_tau = TRUE. Ignored otherwise.}
  \item{nu_tau}{Prior degrees of freedom for spline coefficient scale parameter tau. Default is 2.}
  \item{lambda_tau}{Scale hyperparameter for spline coefficient scale parameter tau. Default is 1.}
}

\value{A list with entries
\item{beta_train_samples}{An array containing spline basis coefficients for each sample in training set. First dimension is number of basis coefficients, second dimension is number of training set samples, and third dimension is nd, number of MCMC samples saved.}
\item{beta_test_samples}{An array containing spline basis coefficients for each sample in the test set.}
\item{sigma_samples}{Vector of all samples of residual standard deviation sigma, including burn-in}
\item{tau_samples}{If fixed_tau = FALSE, vector of all samples of spline basis scaling parameter tau, including burn-in}
\item{theta_samples}{If fixed_theta = FALSE, matrix containing samples of split index probabilities.}
}


