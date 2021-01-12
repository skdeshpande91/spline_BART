# Generate data using the same setup as Low-Kam et al. (2015)
library(splines)

simulate_curve <- function(X, Z){
  return(Z[2]/(1 + exp(-1.0 * Z[1] * (log(X) - log(Z[3])))))
}

set.seed(104)
n <- 100
p <- 25


ni <- rpois(n, lambda = 20)
X_train <- list()
Phi_train <- list()
Z_train <- matrix(runif(n*p, 1, 10), nrow = n, ncol = p)
Y_train <- list()
y_min <- rep(NA, times = n)
y_max <- rep(NA, times = n)
mu_train <- list()
for(i in 1:n){
  tmp_x <- sort(runif(ni[i], min = 1, max = 10))
  Phi_train[[i]] <- bs(tmp_x, knots = seq(1, 10, length = 25), degree = 3)
  X_train[[i]] <- tmp_x
  mu <- simulate_curve(tmp_x, Z_train[i,])
  mu_train[[i]] <- mu
  Y_train[[i]] <- mu + rnorm(ni[i], mean = 0, sd = 1) * 0.2
  y_min[i] <- min(mu)
  y_max[i] <- max(mu)
}

D <- ncol(Phi_train[[1]])

cutpoints <- list()
for(j in 1:p) cutpoints[[j]] <- seq(1, 10, length = 1000)


save(X_train, Y_train, Z_train, Phi_train, mu_train, D, n, p, ni, cutpoints, file = "data/sim1_data.RData")
######################################
# Make an example plot of the data
#######################################
x_seq <- seq(from = 1, to = 10, by = 0.01)


png("writing/figures/low_kam_sim_data.png", width = 4.5, height = 4.5, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(1, type = "n", xlim = c(0, 10), ylim = c(0, 10), xlab = "T",ylab = "q\''", main = "Simulated data")

for(i in 1:n){
  lines(x_seq, simulate_curve(x_seq, Z_train[i,]), col = 'lightgray', lwd = 0.3)
}
lines(x_seq, simulate_curve(x_seq, Z_train[10,]), col = 'purple')
points(X_train[[10]], Y_train[[10]], pch = 16, cex = 0.5, col = 'purple')

lines(x_seq, simulate_curve(x_seq, Z_train[100,]), col = 'blue')
points(X_train[[100]], Y_train[[100]], pch = 16, cex = 0.5, col = 'blue')

lines(x_seq, simulate_curve(x_seq, Z_train[92,]), col = 'red')
points(X_train[[92]], Y_train[[92]], pch = 16, cex = 0.5, col = 'red')

lines(x_seq, simulate_curve(x_seq, Z_train[28,]), col = 'green')
points(X_train[[28]], Y_train[[28]], pch = 16, cex = 0.5, col= 'green')

lines(x_seq, simulate_curve(x_seq, Z_train[37,]), col = 'black')
points(X_train[[37]], Y_train[[37]], pch = 16, cex = 0.5, col = 'black')

lines(x_seq, simulate_curve(x_seq, Z_train[50,]), col = 'orange')
points(X_train[[50]], Y_train[[50]], pch = 16, cex = 0.5, col = 'orange')

dev.off()