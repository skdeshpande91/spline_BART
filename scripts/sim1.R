library(splineBART)

load("data/sim1_data.RData")

set.seed(129)
test_index <- sort(sample(1:length(Y_all), size = 10))
train_index <- sort((1:length(Y_all))[-test_index])

Z_new <- Z_all

Y_train <- Y_all[train_index]
Y_test <- Y_all[test_index]

X_train <- X_all[train_index]
X_test <- X_all[test_index]

Z_train <- Z_new[train_index,]
Z_test <- matrix(Z_new[test_index,], nrow = length(test_index))

n_train <- length(train_index)

###########################
# Get knots
###########################
# for simplicity, we'll truncate to integer values for min/max but this really isn't necessary


n_knots <- 35
intercept <- TRUE
spline_degree <- 3

x_max <- ceiling(max(unlist(X_all)))
x_min <- floor(min(unlist(X_all)))
knot_seq <- seq(x_min, x_max, length = n_knots)


###########################
# Compute Phi_train
###########################
Phi_train <- list()
for(i in 1:n_train){
  Phi_train[[i]] <- bs(X_train[[i]], knots = seq(x_min, x_max, length = n_knots), degree = spline_degree, intercept = intercept)
}
if(length(unique(sapply(Phi_train, FUN= ncol))) != 1){
  stop("Not all elements of Phi_train have same number of elements")
}
D <- ncol(Phi_train[[1]])



###########
# Get a RW1 "precision" matrix
##########

tmp <- get_precision(D, K_ord = 1)
K1 <- tmp$K
log_det_K1 <- tmp$log_det_K
K1_ord <- 1 # D - K1_ord is rank of K1

K1_adj <- K1
K1_adj[1,1] <- K1[1,1] + 1e-3
K1_adj[D,D] <- K1[D,D] + 1e-3
K1_adj_ord <- 0
log_det_K1_adj <- determinant(K1_adj, logarithm = TRUE)$modulus

#############
# run BART
#############

rw1_chain1 <- splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints, tau = 1, K = K1, 
                         log_det_K = log_det_K1, K_ord = 1)
rw1_chain2 <- splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints, tau = 1, K = K1, 
               log_det_K = log_det_K1, K_ord = 1)

adj1_chain1 <- splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints, tau = 1, K = K1_adj, 
                         log_det_K = log_det_K1_adj, K_ord = 0)
adj1_chain2 <- splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints, tau = 1, K = K1_adj, 
                         log_det_K = log_det_K1_adj, K_ord = 0)


id_chain1 <- splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints, tau = 1, K = diag(D), 
                        log_det_K = 0, K_ord = 0)
id_chain2 <- splineBART(Y_train, Phi_train, Z_train, Z_test, cutpoints, tau = 1, K = diag(D), 
                        log_det_K = 0, K_ord = 0)
############
# a few traceplots
############


par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,3))

plot(rw1_chain1$beta_test_samples[1,1,], type = "l", main = "Intercept- test 1", ylab = "beta")
lines(rw1_chain2$beta_train_samples[1,1,], col = 'red')

plot(rw1_chain1$beta_test_samples[1,4,], type = "l", main = "Intercept- test 4", ylab = "beta")
lines(rw1_chain2$beta_test_samples[1,4,], col = 'red')


plot(rw1_chain1$beta_test_samples[31,9,], type = "l", main = "beta31- test 9", ylab = "beta")
lines(rw1_chain2$beta_test_samples[31,9,], col = 'red')



par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(1,3))

plot(adj1_chain1$beta_test_samples[1,1,], type = "l", main = "Intercept- test 1", ylab = "beta")
lines(adj1_chain2$beta_train_samples[1,1,], col = 'red')

plot(adj1_chain1$beta_test_samples[1,4,], type = "l", main = "Intercept- test 4", ylab = "beta")
lines(adj1_chain2$beta_test_samples[1,4,], col = 'red')


plot(adj1_chain1$beta_test_samples[31,9,], type = "l", main = "beta31- test 9", ylab = "beta")
lines(adj1_chain2$beta_test_samples[31,9,], col = 'red')


################
# Now start plotting stuff

x <- seq(from = 1, to = 10, by = 0.1)
Phi_test <- bs(x, knots = seq(x_min, x_max, length = n_knots), degree = spline_degree, intercept = intercept)

rw1_post_summ <- summarize_posterior(rw1_chain1, rw1_chain2, Phi_test)
adj1_post_summ <- summarize_posterior(adj1_chain1, adj1_chain2, Phi_test)
id_post_summ <- summarize_posterior(id_chain1, id_chain2, Phi_test)

train_col_list <- c("purple", "red", "blue")
test_col_list <- c("orange", "green", "cyan")

sim1_curve <- function(X, Z){
  return(Z[2]/(1 + exp(-1.0 * Z[1] * (log(X) - log(Z[3])))))
}

par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(2,3))

for(i in c(1, 29, 60)){
  plot(1, type = "n", xlim = c(1, 10), ylim = c(-2, 10), xlab = "T",ylab = "Response", main = paste("Curve recovery -- Train", i))
  
  lines(x, sim1_curve(x, Z_train[i,]), col = 'purple')
  lines(x, rw1_post_summ$train$mean[[i]], col = 'purple', lty = 2)
  polygon(c(x, rev(x)), c(rw1_post_summ$train$L95[[i]], rev(rw1_post_summ$train$U95[[i]])),
          col = rgb(1, 0, 1, 1/5), border = NA)
  legend("topleft", legend = c("True", "Post. Mean"), lty = c(1,2), bty = "n")
}

for(i in c(1, 6, 10)){
  plot(1, type = "n", xlim = c(1, 10), ylim = c(-2, 10), xlab = "T",ylab = "Response", main = paste("Curve recovery -- Test", i))
  
  lines(x, sim1_curve(x, Z_test[i,]), col = 'blue')
  lines(x, rw1_post_summ$test$mean[[i]], col = 'blue', lty = 2)
  polygon(c(x, rev(x)), c(rw1_post_summ$test$L95[[i]], rev(rw1_post_summ$test$U95[[i]])),
          col = rgb(0, 0, 1, 1/5), border = NA)
  legend("topleft", legend = c("True", "Post. Mean"), lty = c(1,2), bty = "n")
}


par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(2,3))

for(i in c(1, 29, 60)){
  plot(1, type = "n", xlim = c(1, 10), ylim = c(-2, 10), xlab = "T",ylab = "Response", main = paste("Curve recovery -- Train", i))
  
  lines(x, sim1_curve(x, Z_train[i,]), col = 'purple')
  lines(x, adj1_post_summ$train$mean[[i]], col = 'purple', lty = 2)
  polygon(c(x, rev(x)), c(adj1_post_summ$train$L95[[i]], rev(adj1_post_summ$train$U95[[i]])),
          col = rgb(1, 0, 1, 1/5), border = NA)
  legend("topleft", legend = c("True", "Post. Mean"), lty = c(1,2), bty = "n")
}

for(i in c(1, 6, 10)){
  plot(1, type = "n", xlim = c(1, 10), ylim = c(-2, 10), xlab = "T",ylab = "Response", main = paste("Curve recovery -- Test", i))
  
  lines(x, sim1_curve(x, Z_test[i,]), col = 'blue')
  lines(x, adj1_post_summ$test$mean[[i]], col = 'blue', lty = 2)
  polygon(c(x, rev(x)), c(adj1_post_summ$test$L95[[i]], rev(adj1_post_summ$test$U95[[i]])),
          col = rgb(0, 0, 1, 1/5), border = NA)
  legend("topleft", legend = c("True", "Post. Mean"), lty = c(1,2), bty = "n")
}

par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0), mfrow = c(2,3))

for(i in c(1, 29, 60)){
  plot(1, type = "n", xlim = c(1, 10), ylim = c(-2, 10), xlab = "T",ylab = "Response", main = paste("Curve recovery -- Train", i))
  
  lines(x, sim1_curve(x, Z_train[i,]), col = 'purple')
  lines(x, id_post_summ$train$mean[[i]], col = 'purple', lty = 2)
  polygon(c(x, rev(x)), c(id_post_summ$train$L95[[i]], rev(id_post_summ$train$U95[[i]])),
          col = rgb(1, 0, 1, 1/5), border = NA)
  legend("topleft", legend = c("True", "Post. Mean"), lty = c(1,2), bty = "n")
}

for(i in c(1, 6, 10)){
  plot(1, type = "n", xlim = c(1, 10), ylim = c(-2, 10), xlab = "T",ylab = "Response", main = paste("Curve recovery -- Test", i))
  
  lines(x, sim1_curve(x, Z_test[i,]), col = 'blue')
  lines(x, id_post_summ$test$mean[[i]], col = 'blue', lty = 2)
  polygon(c(x, rev(x)), c(id_post_summ$test$L95[[i]], rev(id_post_summ$test$U95[[i]])),
          col = rgb(0, 0, 1, 1/5), border = NA)
  legend("topleft", legend = c("True", "Post. Mean"), lty = c(1,2), bty = "n")
}
