library(Rcpp)
library(RcppArmadillo)
library(splines)

set.seed(104)
n <- 100
p <- 25


ni <- rpois(n, lambda = 10)
x_train <- list()
phi_train <- list()
tphi_rf <- list()
Z_train <- matrix(runif(n*p, 0, 1), nrow = n, ncol = p)
for(i in 1:n){
  x_train[[i]] <- sort(runif(ni[i], min = 0, max = 40)) # make up a maximum temp differential of 40
  phi <- bs(x_train[[i]], knots = seq(0, 40, length = 50), degree = 3)
  phi_train[[i]] <- phi
  rf <- rnorm(ni[i], mean = 0, sd = 1)
  tphi_rf[[i]] <- t(phi) %*% rf
}

D <- ncol(phi_train[[1]])
############
Delta_1 <- matrix(0, nrow = D-1, ncol = D)
for(d in 1:(D-1)) Delta_1[d, c(d, d+1)] <- c(-1, 1)
K1 <- t(Delta_1) %*% Delta_1

K1_eval <- eigen(K1)$values
log_det_K <- sum(log(K1_eval[1:(D-1)]))
K_ord <- 1


cutpoints <- list()
for(j in 1:p) cutpoints[[j]] <- seq(0, 1, length = 5000)


#############
tau <- 1
sigma <- 1

sourceCpp("splineBART/src/test_lil.cpp")

set.seed(12345)


test <- test_lil(Phi_train = phi_train,
                 Z_train = Z_train,
                 cutpoints = cutpoints,
                 D = D, K = K1, log_det_K = log_det_K, K_ord = K_ord, tau = tau, 
                 tPhi_rf = tphi_rf, sigma = sigma)
test$leaf_ix


########
# Compute lil by hand and compare to the computation in Rcpp
#######
# compute P, m, 

P <- array(0, dim = c(D,D,length(test$lil)))
m <- matrix(0, nrow = D, ncol = length(test$lil))

for(l in 1:length(test$lil)) P[,,l] <- 1.0/(tau * tau) * K1

for(i in 1:n){
  P[,,test$leaf_ix[i]] <- P[,,test$leaf_ix[i]] + 1.0/(sigma * sigma) * t(phi_train[[i]]) %*% phi_train[[i]]
  m[,test$leaf_ix[i]] <- m[,test$leaf_ix[i]] + 1.0/(sigma * sigma) * tphi_rf[[i]]
}
lil_R <- rep(NA, times = length(test$lil))
for(l in 1:length(test$lil)){
  P_inv <- solve(P[,,l])
  log_det <- as.numeric(determinant(P[,,l], logarithm = TRUE)$modulus)
  quad_form <- as.numeric(t(m[,l]) %*% P_inv %*% m[,l])
  lil_R[l] <- -0.5 * log_det + 0.5 * quad_form 
}

if(max(abs(lil_R - test$lil)) > 1e-12){
  stop("lil in R not close enough to lil in C++")
}
