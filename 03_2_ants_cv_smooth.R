#' ---
#' title: "Ant data: smoothing spline model"
#' author: Brett Melbourne
#' date: 29 Jan 2024
#' output:
#'     github_document
#' ---

#' Investigate cross-validation with the ants data and a smoothing-spline model

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)

#' Forest ant data:

forest_ants <- read.csv("data/ants.csv") |> 
    filter(habitat=="forest")

#' Example of a smoothing spline model. Try running this next block of code to
#' visualize the model predictions for different values of `df`. Here is df=7.

smooth_trained <- smooth.spline(forest_ants$latitude, forest_ants$richness, df=7)
grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
preds <- data.frame(predict(smooth_trained, x=grid_latitude))
forest_ants |> 
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=x, y=y)) +
    coord_cartesian(ylim=c(0,20))

#' Using `predict` to ask for predictions from the fitted smoothing spline
#' model.

predict(smooth_trained, x=43.2)
predict(smooth_trained, x=forest_ants$latitude)
predict(smooth_trained, x=seq(41, 45, by=0.5))

#' LOOCV (since it's a small dataset)
#' LOOCV is deterministic for this model.

# Function to divide a data set into random partitions for cross-validation
# n:       length of dataset (scalar, integer)
# k:       number of partitions (scalar, integer)
# return:  partition labels (vector, integer)
# 
random_partitions <- function(n, k) {
    min_n <- floor(n / k)
    extras <- n - k * min_n
    labels <- c(rep(1:k, each=min_n),rep(seq_len(extras)))
    partitions <- sample(labels, n)
    return(partitions)
}

# Function to perform k-fold CV for a smoothing spline on ants data
# k:       number of partitions (scalar, integer)
# df:      degrees of freedom in smoothing spline (scalar, integer)
# return:  CV error as MSE (scalar, numeric)
#
cv_smooth_ants <- function(k, df) {
    forest_ants$partition <- random_partitions(nrow(forest_ants), k)
    e <- rep(NA, k)
    for ( i in 1:k ) {
        test_data <- subset(forest_ants, partition == i)
        train_data <- subset(forest_ants, partition != i)
        smooth_trained <- smooth.spline(train_data$latitude, train_data$richness, df=df)
        pred_richness <- predict(smooth_trained, test_data$latitude)$y
        e[i] <- mean((test_data$richness - pred_richness) ^ 2)
    }
    cv_error <- mean(e)
    return(cv_error)
}

#' Test/use the function (LOOCV mode)

cv_smooth_ants(k=nrow(forest_ants), df=7)

#' Explore a grid of values for df (k is always 22 for LOOCV)

grid <- expand.grid(k=nrow(forest_ants), df=2:16)
grid

cv_error <- rep(NA, nrow(grid))
for ( i in 1:nrow(grid) ) {
    cv_error[i] <- cv_smooth_ants(grid$k[i], grid$df[i])
}
result <- cbind(grid, cv_error)

#' Plot the result.

result |>
    ggplot() +
    geom_line(aes(x=df, y=cv_error)) +
    labs(title="LOOCV")

#' We see that MSE prediction error (cv_error) increases dramatically for df
#' beyond 8 or so.

result |> 
    ggplot() +
    geom_line(aes(x=df, y=cv_error)) +
    coord_cartesian(xlim=c(2,8), ylim=c(12,18)) +
    labs(title="LOOCV")

#' Table of results

result

#' LOOCV (k=22) identifies df=3 as the best performing model. Compared to the
#' polynomial we had from before, this model has slightly better prediction
#' accuracy.
#' 

#' | Model              |   LOOCV   |
#' |--------------------|-----------|
#' | Polynomial 2       |   12.88   |
#' | Smoothing spline 3 |   12.52   |
