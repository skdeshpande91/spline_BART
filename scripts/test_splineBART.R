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



sourceCpp("splineBART/src/spline_bart_fixed.cpp")

Z_test <- Z_train

fit_time <- system.time(
  test <- spline_bart_fixed(Y_train = std_Y_train, Phi_train = Phi_train,
                          Z_train = Z_train, Z_test = Z_test,
                          cutpoints = cutpoints,
                          alpha = alpha, beta = beta, 
                          K = K, log_det_K = log_det_K, K_ord = K_ord, tau = tau, 
                          nu_sigma = nu_sigma, lambda_sigma = lambda_sigma, 
                          M = 200, nd = 1000, burn = 250,
                          verbose = TRUE, print_every = 50, debug = FALSE))
###################################
beta_hat_mean <- apply(test$beta_train_samples, MARGIN = c(1,2), FUN = mean)
# beta_hat_mean[,i] is posterior mean of beta for subject i

mu_hat_mean <- list()
for(i in 1:n){
  mu_hat_mean[[i]] <- y_mean + y_sd * Phi_train[[i]] %*% beta_hat_mean[,i]
}

x_seq <- seq(from = 1, to = 10, by = 0.01)
Phi_test <- bs(x_seq, knots = seq(from = 1, to = 10, length = 25), degree = 3)
mu_test <- matrix(nrow = length(x_seq), ncol = n)
for(i in 1:n){
  mu_test[,i] <- y_mean + y_sd * Phi_test %*% beta_hat_mean[,i]
}

simulate_curve <- function(X, Z){
  return(Z[2]/(1 + exp(-1.0 * Z[1] * (log(X) - log(Z[3])))))
}


png("writing/figures/sim1_train_fit.png", width = 8, height = 4, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,2))
plot(1, type = "n", xlim = c(-0.1, 10), ylim = c(-0.1, 10), 
     main = "Function evaluations", 
     xlab = "Actual evaluations", ylab = "Post. mean evaluations")
points(unlist(mu_train), unlist(mu_hat_mean), pch = 16, cex = 0.5)
abline(a = 0, b = 1, col = 'red')
text(x = 2, y = 9, labels = paste("corr. =", round(cor(unlist(mu_train), unlist(mu_hat_mean)), digits = 3)))


plot(1, type = "n", xlim = c(0, 10), ylim = c(0, 10), xlab = "T",ylab = "q\''", main = "Curve recovery")

lines(x_seq, simulate_curve(x_seq, Z_train[10,]), col = 'purple')
lines(x_seq, mu_test[,10], col = 'purple', lty = 3)

lines(x_seq, simulate_curve(x_seq, Z_train[100,]), col = 'blue')
lines(x_seq, mu_test[,100], col = 'blue', lty = 3)

lines(x_seq, simulate_curve(x_seq, Z_train[92,]), col = 'red')
lines(x_seq, mu_test[,92], col = 'red', lty = 3)

lines(x_seq, simulate_curve(x_seq, Z_train[28,]), col = 'green')
lines(x_seq, mu_test[,28], col = 'green', lty = 3)

lines(x_seq, simulate_curve(x_seq, Z_train[37,]), col = 'black')
lines(x_seq, mu_test[,37], col = 'black', lty = 3)

lines(x_seq, simulate_curve(x_seq, Z_train[50,]), col = 'orange')
lines(x_seq, mu_test[,50], col = 'orange', lty = 3)
legend("topleft", legend = c("Actual", "Post. mean"), lty = c(1,3), bty = FALSE, cex = 0.8)
dev.off()



