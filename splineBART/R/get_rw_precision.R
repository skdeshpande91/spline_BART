get_rw_precision <- function(D, rw_order){
  if(!rw_order %in% c(1,2)){
    stop("[get_precision]: Currently supports only 1st and 2nd order random-walk priors (rw_order = 1 or 2)")
  }
  if(rw_order > D){
    stop("[get_precision]: D must be greater than rw_order")
  }
  
  if(rw_order == 1){
    diff_mat <- matrix(0, nrow = D - 1, ncol = D)
    for(i in 1:(D-1)) diff_mat[i,c(i, i+1)] <- c(-1,1)
    # https://inla.r-inla-download.org/r-inla.org/doc/latent/rw1.pdf
  }
  if(rw_order == 2){
    diff_mat <- matrix(0, nrow = D - 2, ncol = D)
    for(i in 1:(D-2)) diff_mat[i,c(i,i+1,i+2)] <- c(1, -2, 1)
    # https://inla.r-inla-download.org/r-inla.org/doc/latent/rw2.pdf
  }

  K <- t(diff_mat) %*% diff_mat
  K_eigval <- eigen(K)$values
  rank_K <- D - rw_order
  
  
  log_det_K <- sum(log(K_eigval[1:(rank_K)]))
  return(list(K = K, log_det_K = log_det_K, rank_K = rank_K))
}
