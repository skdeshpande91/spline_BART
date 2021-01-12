library(Rcpp)
library(RcppArmadillo)
library(splines)

sourceCpp("splineBART/src/test_fit.cpp")

############
set.seed(104)
n <- 100
p <- 25


ni <- rpois(n, lambda = 10)
x_train <- list()
phi_train <- list()
Z_train <- matrix(runif(n*p, 0, 1), nrow = n, ncol = p)
Y_train <- list()
for(i in 1:n){
  Y_train[[i]] <- rnorm(ni[i], 0, 1)
  x_train[[i]] <- sort(runif(ni[i], min = 0, max = 40)) # make up a maximum temp differential of 40
  phi_train[[i]] <- bs(x_train[[i]], knots = seq(0, 40, length = 50), degree = 3)
}

D <- ncol(phi_train[[1]])


cutpoints <- list()
for(j in 1:p) cutpoints[[j]] <- seq(0, 1, length = 5000)

set.seed(12345)
test <- test_tree_fit(Y_train, phi_train, Z_train, cutpoints, D)


# in class, getbots returns bottom nodes in a left-to-right order (even if leafs are at different depths)
# this is because it always recurses down left children first and then right children
# so to test that we're fitting things right, we'll create a leaf index in R that obeys the order
# in which the class member getbots returns leaf indices

# The tree has 3 bottom nodes
# node 1: if Z[,19] < cutpoints[[19]][4379] go to node 2
# node 1: if Z[,19] >= cutpoints[[19]][4379] go to node 3 (a bottom node)
# node 2: if Z[,23] < cutpoints[[23]][2283] go to node 4
# node 2: if Z[,23] >= cutpoints[[23]][2283] go to node 5


leaf_ix <- rep(NA, times = n)

leaf_ix[which(Z_train[,19] < cutpoints[[19]][[4379]] & Z_train[,23] < cutpoints[[23]][[2283]])] <- 1
leaf_ix[which(Z_train[,19] < cutpoints[[19]][[4379]] & Z_train[,23] >= cutpoints[[23]][[2283]])] <- 2
leaf_ix[which(Z_train[,19] >= cutpoints[[19]][[4379]])] <- 3

my_beta <- test$unik_beta[,leaf_ix]
my_phi_beta <- list()
my_tphi_phi_beta <- matrix(nrow = D, ncol = n)
for(i in 1:n){
  my_phi_beta[[i]] <- phi_train[[i]] %*% my_beta[,i]
  my_tphi_phi_beta[,i] <- t(phi_train[[i]]) %*% phi_train[[i]] %*% my_beta[,i]
}
if(!identical(my_beta, test$beta)){
  stop("error in getting betas")
}
if(!identical(my_phi_beta, test$phi_beta)){
  stop("error in computing my_phi_beta!")
}
if(!identical(my_tphi_phi_beta, test$tphi_phi_beta)){
  stop("error in computing my_tphi_phi_beta!")
}

###############################
set.seed(1042021)
test2 <- test_tree_fit(Y_train, phi_train, Z_train, cutpoints, D)
# node 1: Z[,24] < cutpoints[[24]][3518] go to node 2 (bottom node)
# node 1: Z[,24] >= cupoints[[24]][3518] go to node 3
# node 3: Z[,13] < cutpoints[[13]][1875] go to node 6 (bottom node)
# node 7: Z[,13] >= cutpoints[[13]][1875] go to node 7 (bottom node)

leaf_ix2 <- rep(NA, times = n)
leaf_ix2[which(Z_train[,24] < cutpoints[[24]][3518])] <- 1
leaf_ix2[which(Z_train[,24] >= cutpoints[[24]][3518] & Z_train[,13] < cutpoints[[13]][1875])] <- 2
leaf_ix2[which(Z_train[,24] >= cutpoints[[24]][3518] & Z_train[,13] >= cutpoints[[13]][1875])] <- 3

my_beta2 <- test2$unik_beta[,leaf_ix2]
my_phi_beta2 <- list()
my_tphi_phi_beta2 <- matrix(nrow = D, ncol = n)
for(i in 1:n){
  my_phi_beta2[[i]] <- phi_train[[i]] %*% my_beta2[,i]
  my_tphi_phi_beta2[,i] <- t(phi_train[[i]]) %*% phi_train[[i]] %*% my_beta2[,i]
}
if(!identical(my_beta2, test2$beta)){
  stop("error in getting betas")
}
if(!identical(my_phi_beta2, test2$phi_beta)){
  stop("error in computing my_phi_beta!")
}
if(!identical(my_tphi_phi_beta2, test2$tphi_phi_beta)){
  stop("error in computing my_tphi_phi_beta!")
}
