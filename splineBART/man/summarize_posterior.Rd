\name{summarize_posterior}
\alias{summarize_posterior}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Summarize posterior distribution of regression function and posterior predictive distribution.
}
\description{
Given a list of spline basis design matrices, a list of samples of spline basis coefficients, and (optionally) samples of the residual standard deviation, returns posterior mean and quantiles of the regression function E[Y|X,Z] and posterior predictive distribution.
}
\usage{
summarize_posterior(beta_list, Phi_list, y_mean, y_sd, sigma_list, probs)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{beta_list}{List of beta samples. Each list element must be an 3-dimensional array and must all be of the same dimension}
  \item{Phi_list}{List of spline basis design matrices. Length of Phi_list must be equal to second dimension of elements of beta_list.}
  \item{y_mean}{\code{splineBART} runs with standardized outcomes, so posterior summaries must be transformed back to original scale. Typically \code{y_mean} is equal to mean of training set observations.}
  \item{y_sd}{\code{splineBART} runs with standardized outcomes, so posterior summaries must be transformed back to original scale. Typically \code{y_sd} is equal to the standard deviation of testing set observations.}
  \item{sigma_list}{Optional list containing post-burn in samples of residual standard deviation sigma. The lists \code{sigma_list} and \code{beta_list} must be of the same length.}
  \item{probs}{Vector of probabilities for desired posterior quanties. Default is \code{c(0.025, 0.975).}}
}
\value{A list with entries
\item{fit}{A list of matrices, one for each element of \code{Phi_list}, containing summaries of posterior over regression function. The first column of each matrix is the posterior mean and the subsequent columns are the quantiles specified by \code{probs}.}
\item{ystar}{Analogous to \code{fit} but for the posterior predictive distribution.}
}
