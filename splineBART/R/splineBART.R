splineBART <- function(Y_train,
                       Phi_train,
                       Z_train,
                       Z_test,
                       cutpoints,
                       K,
                       log_det_K,
                       rank_K,
                       M = 50,
                       alpha = 0.95,
                       beta = 2,
                       nu_sigma = 3,
                       sigma_hat = 1,
                       sigquant = 0.9,
                       nd = 1000, burn = 250, 
                       verbose = TRUE, print_every = 50,
                       fixed_theta = FALSE, fixed_tau = FALSE,
                       theta = NULL,
                       a = NULL,
                       b = NULL,
                       N_u = NULL,
                       rho_eta = NULL,
                       tau = NULL, 
                       nu_tau = NULL,
                       lambda_tau = NULL)
{
  
  # check that training data are passed as lists
  if(!is.list(Y_train)) stop("Y_train must be a list of vectors!")
  if(!is.list(Phi_train) ) stop("Phi_train must be lists of matrices!")
  if(!is.matrix(Z_train) | !is.matrix(Z_test)) stop("Z_train & Z_test must be matrices!")
  
  n_train <- length(Y_train) # total number of subjects/units
  p <- ncol(Z_train)
  D <- ncol(Phi_train[[1]])
  # check dimensions
  if(length(Phi_train) != n_train) stop("Y_train and Phi_train must be of same length!")
  if(nrow(Z_train) != n_train) stop("Z_train must have length(Y_train) rows!")
  if(ncol(Z_test) != p) stop("Z_test must have the same number of columns as Z_train!")
  if(!all(sapply(Phi_train, FUN = ncol) == D)) stop("All elemnts of Phi_train must have same number of columns!")
  
  ###########################
  # Standardize Y
  ###########################
  y_mean <- mean(unlist(Y_train))
  y_sd <- sd(unlist(Y_train))
  
  std_Y_train <- list()
  for(i in 1:n_train)  std_Y_train[[i]] <- (Y_train[[i]] - y_mean)/y_sd
  
  ############################
  # Set the lambda_sigma parameter
  ############################
  
  lambda_sigma <- sigma_hat * sigma_hat * qchisq(1 - sigquant, df = nu_sigma)/nu_sigma
  # Compute lambda_sigma
  # come up with a better name: sigquant is what is used by wbart
  # "quantile of prior that rough estimate (sigest) is placed at. Closer quantile is to 1, more aggressive
  # the fit will be as you are putting more weight on residual standard deviations (sigma) *less* than the rough estimate
  # default sigquant <- 0.9
  
  
  ###################
  if(fixed_theta & fixed_tau){
    
    print("Running splineBART with fixed variable split probs. & fixed prior scale")
    
    
    if(is.null(theta)) theta <- rep(1/p, times = p)
    if(is.null(tau) || tau <= 0) stop("If fixed_tau = TRUE, tau must be specified and must be positive")
    
    run_time <-
      system.time(
        fit <- .fixed_theta_fixed_tau(Y_train = std_Y_train,
                                      Phi_train = Phi_train,
                                      Z_train = Z_train,
                                      Z_test = Z_test,
                                      cutpoints = cutpoints,
                                      alpha = alpha, beta = beta,
                                      K = K, log_det_K = log_det_K, rank_K = rank_K,
                                      tau = tau,
                                      nu_sigma = nu_sigma, lambda_sigma = lambda_sigma,
                                      M = M, nd = nd, burn = burn, 
                                      verbose = verbose, print_every = print_every, debug = FALSE))
    fit[["time"]] <- run_time["elapsed"]
  } else if(fixed_theta & !fixed_tau){
    print("Running splineBART with fixed variable split probs. & adaptive prior scale")
    
    if(is.null(theta)) theta <- rep(1/p, times = p)
    if(!is.null(tau)) warning("fixed_tau = FALSE, ignoring the specified tau argument.")
    
    
    
    # For now, default prior on tau^2 ~ IG(1, 1), which is somewhat weakly informative
    if(is.null(nu_tau)) nu_tau <- 2
    if(is.null(lambda_tau)) lambda_tau <- 1
    

    run_time <-
      system.time(
        fit <- .fixed_theta_adapt_tau(Y_train = std_Y_train,
                                      Phi_train = Phi_train,
                                      Z_train = Z_train,
                                      Z_test = Z_test,
                                      cutpoints = cutpoints,
                                      alpha = alpha, beta = beta,
                                      K = K, log_det_K = log_det_K, rank_K = rank_K,
                                      nu_tau = nu_tau, lambda_tau = lambda_tau,
                                      nu_sigma = nu_sigma, lambda_sigma = lambda_sigma,
                                      M = M, nd = nd, burn = burn, 
                                      verbose = verbose, print_every = print_every, debug = FALSE))
    
    fit[["time"]] <- run_time["elapsed"]
  } else if(!fixed_theta & !fixed_tau){
    print("Running splineBART with adaptive variable split probs. & adaptive prior scale")
    
    if(is.null(a)) a <- 1
    if(is.null(b)) b <- p
    if(is.null(rho_eta)) rho_eta <- p
    if(is.null(N_u)) N_u <- 100
    
    
    if(is.null(nu_tau)) nu_tau <- 2
    if(is.null(lambda_tau)) lambda_tau <- 1
    
    run_time <-
      system.time(
        fit <- .adapt_theta_adapt_tau(Y_train = std_Y_train,
                                      Phi_train = Phi_train,
                                      Z_train = Z_train,
                                      Z_test = Z_test,
                                      cutpoints = cutpoints,
                                      alpha = alpha, beta = beta,
                                      K = K, log_det_K = log_det_K, rank_K = rank_K,
                                      nu_tau = nu_tau, lambda_tau = lambda_tau,
                                      nu_sigma = nu_sigma, lambda_sigma = lambda_sigma,
                                      a = a, b = b, N_u = N_u, rho_eta = rho_eta,
                                      M = M, nd = nd, burn = burn, 
                                      verbose = verbose, print_every = print_every, debug = FALSE))
    
    fit[["time"]] <- run_time["elapsed"]
  } else if(!fixed_theta & fixed_tau){
    # hyperparameters for variable split probabilities
    print("Running splineBART with adaptive variable split probs. & fixed prior scale")
    if(is.null(a)) a <- 1
    if(is.null(b)) b <- p
    if(is.null(rho_eta)) rho_eta <- p
    if(is.null(N_u)) N_u <- 100
    
    if(is.null(tau) || tau <= 0) stop("If fixed_tau = TRUE, tau must be specified and must be positive")
    run_time <-
      system.time(
        fit <- .adapt_theta_fixed_tau(Y_train = std_Y_train,
                                      Phi_train = Phi_train,
                                      Z_train = Z_train,
                                      Z_test = Z_test,
                                      cutpoints = cutpoints,
                                      alpha = alpha, beta = beta,
                                      K = K, log_det_K = log_det_K, rank_K = rank_K,
                                      tau = tau,
                                      nu_sigma = nu_sigma, lambda_sigma = lambda_sigma,
                                      a = a, b = b, N_u = N_u, rho_eta = rho_eta,
                                      M = M, nd = nd, burn = burn, 
                                      verbose = verbose, print_every = print_every, debug = FALSE))
    
    
    
  }
  return(fit)
}