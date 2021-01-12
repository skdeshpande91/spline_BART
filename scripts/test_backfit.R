library(splines)
library(Rcpp)
library(RcppArmadillo)


load("data/sim1_data.RData")

############################
# Set tree hyperparameters
############################
alpha <- 0.95
beta <- 2
M <- 200
tau <- 1
############################
# Define difference matrix 
# and "precision matrix"
###########################
diff_mat <- matrix(0, nrow = D-1, ncol = D)
for(i in 1:(D-1)) diff_mat[i,c(i,i+1)] <- c(-1,1)
K <- t(diff_mat) %*% diff_mat
K_eigval <- eigen(K)$values

K_ord <- 1
log_det_K <- sum(log(K_eigval[1:(D-1)]))

##########################
# Standardize Y
##########################
y_mean <- mean(unlist(Y_train))
y_sd <- sd(unlist(Y_train))

std_Y_train <- list()
for(i in 1:n){
  std_Y_train[[i]] <- (Y_train[[i]] - y_mean)/y_sd
}

############################
# Set the sigma parameter
############################

nu_sigma <- 3
lambda_sigma <- qchisq(0.1, df = nu_sigma)/nu_sigma


sourceCpp("splineBART/src/test_backfit.cpp")

test <- test_backfit(std_Y_train, Phi_train, Z_train, cutpoints,
                     alpha, beta, K, log_det_K, K_ord, tau,
                     nu_sigma, lambda_sigma, 200, FALSE)

# check that we've computed the residuals correctly
max_diff <- rep(NA, times = n)
for(i in 1:n){
  tmp_resid <- std_Y_train[[i]] - Phi_train[[i]] %*% test$beta[,i]
  max_diff[i] <- max(abs(tmp_resid - test$rf[[i]]))
}

if(max(max_diff) > 1e-12){
  stop("residual calculation is off by more than 1e-12")
}