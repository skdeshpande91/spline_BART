library(Rcpp)
library(RcppArmadillo)

sourceCpp("splineBART/src/test_lin_alg.cpp")

set.seed(129)
n <- 100
D <- 50
tmp_P <- matrix(rnorm(n*D, 0, 1), nrow = n, ncol = D)
P <- t(tmp_P) %*% tmp_P / n

P_inv <- solve(P)
m <- rnorm(D, 0, 1)

log_det_P <- as.numeric(determinant(P, logarithm = TRUE)$modulus)
quad_form <- t(m) %*% P_inv %*% m
post_mean <- P_inv %*% m

test <- test_lin_alg(P,m)

if(!identical(test$log_det_P, log_det_P)){
  stop("log_det_P mismatch")
}
if(!identical(test$quad_form1, test$quad_form2)){
  stop("quad_form mismatch")
}
if(max(abs(test$post_mean1 - test$post_mean2)) > 1e-12){
  stop("post_mean mismatch")
}
if(abs(test$log_det_L1 - test$log_det_L2) > 1e-12){
  stop("log_det_L mismatch")
}
if(abs(2.0 * test$log_det_L2 - test$log_det_P) > 1e-12){
  stop("2 * log_det_L does not match log_det_P")
}
