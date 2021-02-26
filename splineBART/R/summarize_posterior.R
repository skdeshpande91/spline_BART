summarize_posterior <- function(beta_list,
                                Phi_list,
                                y_mean,
                                y_sd,
                                sigma_list = NULL,
                                probs = c(0.025, 0.975))
{
  if(ncol(unique(sapply(beta_list, FUN = dim), MARGIN = 2)) != 1) stop("All elements of beta_list must have same dimensions")
  n_chains <- length(beta_list)
  D <- dim(beta_list[[1]])[1]
  n <- dim(beta_list[[1]])[2]
  n_samples <- dim(beta_list[[1]])[3]
  
  if(length(Phi_list) != n) stop("Phi_list must contain dim(beta_list[[1]])[2] elements")
  if(!all(sapply(Phi_list, FUN = ncol) == D)) stop("All elements of Phi_list must have dim(beta_list[[1]])[1] columns")
  
  if(!is.null(sigma_list)){
    if(length(sigma_list) != n_chains) stop("beta_list & sigma_list must be of the same length")
    if(!all(sapply(sigma_list, FUN = length) == n_samples)) stop("All elements of sigma_list must have length dim(beta_list[[1]])[3]")
    post_sum <- .summarize_fit_ystar(beta_list, Phi_list, sigma_list, probs, n_samples,y_mean, y_sd)
  } else{
    print("Since samples of sigma were not provided, only summarizing posterior of evaluations of regression function E[Y|X,Z].")
    post_sum <- .summarize_fit(beta_list, Phi_list,probs,n_samples, y_mean, y_sd)
  }
  return(post_sum)
}
