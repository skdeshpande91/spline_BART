library(splineBART)

load("data/sim0_data.RData")
set.seed(224)
sigma <- 0.5

Y_all <- list()
for(i in 1:n) Y_all[[i]] <- mu_all[[i]] + sigma * rnorm(length(mu_all[[i]]), 0, 1)

n_train <- floor(0.8 * n)
n_test <- n - n_train

train_index <- sort(sample(1:n, size = n_train))
test_index <- sort((1:n)[-train_index])

X_train <- X_all[train_index]
Y_train <- Y_all[train_index]
Z_train <- Z_all[train_index,]
mu_train <- mu_all[train_index]

X_test <- X_all[test_index]
Y_test <- Y_all[test_index]
Z_test <- Z_all[test_index,]
mu_test <- mu_all[test_index]

##############
# Create a b-splines basis w/ intercept

n_knots <- 25
spline_degree <- 3
intercept <- TRUE

x_max <- ceiling(max(unlist(X_all)))
x_min <- floor(min(unlist(X_all)))
knot_seq <- seq(x_min, x_max, length = n_knots)

Phi_train <- list()
for(i in 1:n_train){
  Phi_train[[i]] <- bs(X_train[[i]], knots = knot_seq, degree = spline_degree, intercept = intercept)
}

Phi_test <- list()
for(i in 1:n_test){
  Phi_test[[i]] <- bs(X_test[[i]], knots = knot_seq, degree = spline_degree, intercept = intercept)
}


if(length(unique(sapply(Phi_train, FUN= ncol))) != 1){
  stop("Not all elements of Phi_train have same number of elements")
}
D <- ncol(Phi_train[[1]])

###########
# Get a RW1 "precision" matrix
##########

tmp <- get_rw_precision(D, rw_order = 1)
K1 <- tmp$K
log_det_K1 <- tmp$log_det_K
rank_K1 <- tmp$rank_K

K1_adj <- K1
K1_adj[1,1] <- K1[1,1] + 1e-6
K1_adj[D,D] <- K1[D,D] + 1e-6
rank_K1_adj <- D
log_det_K1_adj <- determinant(K1_adj, logarithm = TRUE)$modulus

###########
# Use RW1:

y_sd_train <- sd(unlist(Y_train))
y_mean_train <- mean(unlist(Y_train))
burn <- 500
rw1_chain1 <- splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints,
                         K = K1, log_det_K = log_det_K1, rank_K = rank_K1, M = 25, burn = burn)
rw1_chain2 <- splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints,
                         K = K1, log_det_K = log_det_K1, rank_K = rank_K1, M = 25, burn = burn)

rw1_summ_train <- summarize_posterior(beta_list = list(rw1_chain1$beta_train_samples, rw1_chain2$beta_train_samples),
                                      Phi_list = Phi_train,
                                      sigma_list = list(rw1_chain1$sigma_samples[-(1:burn)], rw1_chain2$sigma_samples[-(1:burn)]),
                                      y_mean = y_mean_train,
                                      y_sd = y_sd_train)
rw1_summ_test <- summarize_posterior(beta_list = list(rw1_chain1$beta_test_samples, rw1_chain2$beta_test_samples),
                                      Phi_list = Phi_test,
                                      sigma_list = list(rw1_chain1$sigma_samples[-(1:burn)], rw1_chain2$sigma_samples[-(1:burn)]),
                                      y_mean = y_mean_train,
                                      y_sd = y_sd_train)

plot(1, type = "n", xlim = range(unlist(mu_train)), ylim = range(unlist(mu_train)))
for(i in 1:n_train){
  points(rw1_summ_train$fit[[i]][,1], mu_train[[i]], pch = 16, cex = 0.5)
}
abline(a = 0, b = 1, col = 'red')

plot(1, type = "n", xlim = range(unlist(mu_test)), ylim = range(unlist(mu_test)))
for(i in 1:n_test){
  points(rw1_summ_test$fit[[i]][,1], mu_test[[i]], pch = 16, cex = 0.5)
}

abline(a = 0, b = 1, col = 'red')
