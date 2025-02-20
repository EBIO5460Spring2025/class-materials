#' ---
#' title: "Stochastic gradient descent training algorithm"
#' author: Brett Melbourne
#' date: 20 Feb 2025
#' output:
#'     github_document
#' ---

#' Stochastic gradient descent illustrated with a simple linear model. The code
#' here is substantially the same as the deterministic version. We are just
#' adding a step to the algorithm to randomly sample the data rows at each
#' iteration.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)

#' Generate data for a linear model with one x variable.

set.seed(7362)
b_0 <- 0
b_1 <- 1
n <- 100 #number of data points
x <- runif(n, -1, 1)
y <- rnorm(length(x), mean=b_0 + b_1 * x, sd=0.25)
ggplot(data.frame(x, y)) +
    geom_point(aes(x, y))


#' Stochastic Gradient descent algorithm:
#' ```
#' set lambda
#' make initial guess for b_0, b_1
#' for many iterations
#'     randomly sample rows (y, x)                <- the new line
#'     find gradient at b_0, b_1
#'     step down: b = b - lambda * gradient(b)
#' print b_0, b_1
#' ```

#' Code this algorithm in R

lambda <- 0.01
b_0 <- -0.5
b_1 <- 1.4
p <- 0.5 #proportion of data to sample
iter <- 10000
for ( i in 1:iter ) {
    
    # randomly sample rows (y, x)                         <- the new line
    row_i <- sample(1:length(y), trunc(p * length(y)))   
    
    # find gradient at b_0, b_1
    y_pred <- b_0 + b_1 * x[row_i]
    r <- y[row_i] - y_pred
    db_1 <- -2 * sum(r * x[row_i]) / n
    db_0 <- -2 * sum(r) / n
    
    # step down
    b_0 <- b_0 - lambda * db_0
    b_1 <- b_1 - lambda * db_1
}
b_0
b_1

#' Compare to the inbuilt algorithm (a numerical solution)

lm(y~x)


#' Animate the algorithm
#+ cache=TRUE, eval=FALSE

grid <- expand.grid(b_0 = seq(-1, 1, 0.05), b_1 = seq(0, 2, 0.05))
mse <- rep(NA, nrow(grid))
for ( i in 1:nrow(grid) ) {
    y_pred <- grid$b_0[i] + grid$b_1[i] * x
    mse[i] <- sum((y - y_pred) ^ 2) / n
}
mse_grid <- cbind(grid, mse)

b_0_vals <- unique(mse_grid$b_0)
mse_b_0 <- rep(NA, length(b_0_vals))
for ( i in 1:length(b_0_vals) ) {
    mse_b_0[i] <- min(mse_grid[mse_grid$b_0 == b_0_vals[i],"mse"])
}

b_1_vals <- unique(mse_grid$b_1)
mse_b_1 <- rep(NA, length(b_1_vals))
for ( i in 1:length(b_1_vals) ) {
    mse_b_1[i] <- min(mse_grid[mse_grid$b_1 == b_1_vals[i],"mse"])
}


# Pause for the specified number of seconds.
pause <- function( secs ) {
    start_time <- proc.time()
    while ( (proc.time() - start_time)["elapsed"] < secs ) {
        #Do nothing
    }
}

# Function to make gradient plots
make_plots <- function() {
    grid_b_0 <- seq(-1, 1, 0.05)
    mse <- rep(NA, length(grid_b_0))
    for ( i in 1:length(grid_b_0) ) {
        y_pred <- grid_b_0[i] + b_1 * x[row_i]
        mse[i] <- sum((y[row_i] - y_pred) ^ 2) / length(row_i)
    }
    plot(b_0_vals, mse_b_0, type="l", ylim=c(0, 0.4),
         main=paste("Iteration", j))
    lines(grid_b_0, mse, col="blue")
    y_pred <- b_0 + b_1 * x[row_i]
    mse <- sum((y[row_i] - y_pred) ^ 2) / length(row_i)
    points(b_0, mse, col="red", pch=19)
    abline(mse - db_0 * b_0, db_0, col="red")
    
    grid_b_1 <- seq(0, 2, 0.05)
    mse <- rep(NA, length(grid_b_1))
    for ( i in 1:length(grid_b_1) ) {
        y_pred <- b_0 + grid_b_1[i] * x[row_i]
        mse[i] <- sum((y[row_i] - y_pred) ^ 2) / length(row_i)
    }
    plot(b_1_vals, mse_b_1, type="l", ylim=c(0, 0.4))
    lines(grid_b_1, mse, col="blue")
    y_pred <- b_0 + b_1 * x[row_i]
    mse <- sum((y[row_i] - y_pred) ^ 2) / length(row_i)
    points(b_1, mse, col="red", pch=19)
    abline(mse - db_1 * b_1, db_1, col="red")
}

# Gradient descent algorithm with gradient plots
par(mfrow=c(1,2))
epoch <- 10
lambda <- 0.01
b_0 <- -0.5
b_1 <- 1.4
p <- 0.5 #proportion of data to sample
iter <- 1000
for ( j in 0:iter ) {

    # randomly sample rows (y, x)
    row_i <- sample(1:length(y), trunc(p * length(y)))   
    
    # find gradient at b_0, b_1
    y_pred <- b_0 + b_1 * x[row_i]
    r <- y[row_i] - y_pred
    db_1 <- -2 * sum(r * x[row_i]) / n
    db_0 <- -2 * sum(r) / n
    
    # Make plots
    if ( (j <= 50 & j %% 5 == 0 ) | j %% epoch == 0 ) {
        make_plots()
        pause(1)
    }
    
    # step down
    b_0 <- b_0 - lambda * db_0
    b_1 <- b_1 - lambda * db_1
    
}
