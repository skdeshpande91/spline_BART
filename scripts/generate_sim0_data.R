# Replicate the Low-Kam et al (2015) simulation exactly
library(splines)

sim0_curve <- function(X, Z){
  return(Z[2]/(1 + exp(-1.0 * Z[1] * (log(X) - log(Z[3])))))
}

n <- 50
p <- 8

set.seed(129)
Z_all <- matrix(nrow = n, ncol = p)
Z_all[,4:p] <- runif(n*(p-3), min = 1, max = 10)

Z_all[1:30,1] <- 0
Z_all[1:30,2] <- 2
Z_all[1:30,3] <- runif(30, min = 1, max = 10)

Z_all[31:50,1] <- runif(20, min = 1, max = 10)
Z_all[31:50,2] <- runif(20, min = 1, max = 10)

Z_all[31:50,3] <- (Z_all[31:50,2] - 1)^(1/(Z_all[31:50,1]))

test_z3 <- rep(NA, times = n)
for(i in 1:n) test_z3[i] <- sim0_curve(1, Z_all[i,])
if(!all(abs(test_z3 - 1) < 1e-12)) stop("Mistake in setting Z[,3]!")

X_all <- list()
mu_all <- list()
#Y_all <- list()

for(i in 1:n){
  X_all[[i]] <- 1:11
  mu <- sim0_curve(X_all[[i]], Z_all[i,])
  mu_all[[i]] <- mu
  #Y_all[[i]] <- mu + rnorm(length(X_all[[i]]), mean = 0, sd = 1)
}


cutpoints <- list()
for(j in 1:p) cutpoints[[j]] <- seq(0, 10, length = 1000)

save(X_all, mu_all, Z_all, n, p, sim0_curve, cutpoints, file = "data/sim0_data.RData")
