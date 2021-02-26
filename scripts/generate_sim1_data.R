library(splines)

sim1_curve <- function(X, Z){
  return(Z[2]/(1 + exp(-1.0 * Z[1] * (log(X) - log(Z[3])))))
}

set.seed(104)
n <- 100
p <- 25


ni <- rpois(n, lambda = 20)
X_all <- list()
Phi_all <- list()
Z_all <- matrix(runif(n*p, 1, 10), nrow = n, ncol = p)
Y_all <- list()
y_min <- rep(NA, times = n)
y_max <- rep(NA, times = n)
mu_all <- list()
for(i in 1:n){
  tmp_x <- sort(runif(ni[i], min = 1, max = 10))
  Phi_all[[i]] <- bs(tmp_x, knots = seq(1, 10, length = 25), degree = 3)
  X_all[[i]] <- tmp_x
  mu <- sim1_curve(tmp_x, Z_all[i,])
  mu_all[[i]] <- mu
  Y_all[[i]] <- mu + rnorm(ni[i], mean = 0, sd = 1) * 0.2
  y_min[i] <- min(mu)
  y_max[i] <- max(mu)
}

D <- ncol(Phi_all[[1]])

cutpoints <- list()
for(j in 1:p) cutpoints[[j]] <- seq(1, 10, length = 1000)

save(X_all, Y_all, Z_all, Phi_all, mu_all, D, n, p, ni, cutpoints, file = "data/sim1_data.RData")
######################################
# Make an example plot of the data
#######################################
x_seq <- seq(from = 1, to = 10, by = 0.01)


png("writing/figures/low_kam_sim_data.png", width = 4.5, height = 4.5, units = "in", res = 300)
par(mar = c(3,3,2,1), mgp = c(1.8, 0.5, 0))
plot(1, type = "n", xlim = c(0, 10), ylim = c(0, 10), xlab = "T",ylab = "q\''", main = "Simulated data")

for(i in 1:n){
  lines(x_seq, sim1_curve(x_seq, Z_all[i,]), col = 'lightgray', lwd = 0.3)
}
lines(x_seq, sim1_curve(x_seq, Z_all[10,]), col = 'purple')
points(X_all[[10]], Y_all[[10]], pch = 16, cex = 0.5, col = 'purple')

lines(x_seq, sim1_curve(x_seq, Z_all[100,]), col = 'blue')
points(X_all[[100]], Y_all[[100]], pch = 16, cex = 0.5, col = 'blue')

lines(x_seq, sim1_curve(x_seq, Z_all[92,]), col = 'red')
points(X_all[[92]], Y_all[[92]], pch = 16, cex = 0.5, col = 'red')

lines(x_seq, sim1_curve(x_seq, Z_all[28,]), col = 'green')
points(X_all[[28]], Y_all[[28]], pch = 16, cex = 0.5, col= 'green')

lines(x_seq, sim1_curve(x_seq, Z_all[37,]), col = 'black')
points(X_all[[37]], Y_all[[37]], pch = 16, cex = 0.5, col = 'black')

lines(x_seq, sim1_curve(x_seq, Z_all[50,]), col = 'orange')
points(X_all[[50]], Y_all[[50]], pch = 16, cex = 0.5, col = 'orange')

dev.off()