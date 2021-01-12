library(Rcpp)
library(RcppArmadillo)

sourceCpp("src/test.cpp")



Phi_train <- list(matrix(1:10, 2, 5), matrix(1:50, 10, 5))
tPhi_Phi_train <- list()
for(i in 1:2) tPhi_Phi_train[[i]] <- t(Phi_train[[i]]) %*% Phi_train[[i]]
tPhi_Y_train <- list(1:5, 6:55)
Z_train <- matrix(1:20, 2, 10)

test <- test_input(Phi_train, tPhi_Phi_train, tPhi_Y_train, Z_train, D = 50)



test <- test_pointer(Phi_train, tPhiY, Z_train)

test <- test_input(Phi_train, tPhiY, Z_train)


identical(Phi_train[[1]], test[[1]])
