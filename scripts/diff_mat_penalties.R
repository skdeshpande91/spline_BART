# Second difference matrix with 6 elements
D2 <- matrix(0, nrow = 8, ncol = 10)
for(i in 1:8){
  D2[i,i:(i+2)] <- c(1, -2, 1)
}

K2 <- t(D2) %*% D2

K2_adj <- K2 + diag(c(1e-6, 1e-6, rep(0, times = 6), 1e-6, 1e-6))
eigen(K2_adj)$values


d <- 25
D1 <- matrix(0, nrow = d-1, ncol = d)
for(i in 1:(d-1)) D1[i,c(i,i+1)] <- c(-1,1)
K1 <- t(D1) %*% D1
eta <- 1e-4
K1_adj <- K1+ diag(c(eta, rep(0, times = d-2), eta))

eigen(K1_adj)$values

eta <- 1e-6
