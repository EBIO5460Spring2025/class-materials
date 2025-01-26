#' ---
#' title: "Ant data: k-fold cross validation"
#' author: Brett Melbourne
#' date: 23 Jan 2025
#' output:
#'     github_document
#' ---

#' Explore the cross-validation **inference algorithm** from scratch with the
#' ants data and a polynomial model. Our goal is to predict richness of forest
#' ants from latitude. What order of a polynomial **model algorithm** gives the
#' most accurate predictions?

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
library(tidyr) #for pivot_longer()

#' Ant data:

ants <- read.csv("data/ants.csv")
head(ants)

#' Forest ant data:

forest_ants <- ants |> 
    filter(habitat=="forest")

forest_ants |>
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    ylim(0,20)

#' Here's one way we could code a 3rd order polynomial by first creating new
#' variables for the quadratic (squared) and cubic (cubed) terms, and using R's
#' model formula syntax to train the model by minimizing the SSQ with the
#' function `lm`.

forest_ants$latitude_2 <- forest_ants$latitude ^ 2
forest_ants$latitude_3 <- forest_ants$latitude ^ 3
head(forest_ants)
lm(richness ~ latitude + latitude_2 + latitude_3, data=forest_ants)

#' A model formula provides a shorthand notation for (mostly) linear models,
#' e.g. `y ~ x + z` is shorthand for the model:
#' 
#' $$
#' y = \beta_0 + \beta_1 x + \beta_2 z
#' $$
#' 

#' Here's another way to code the same model that eliminates the need to create
#' new variables for higher order terms.

lm(richness ~ latitude + I(latitude^2) + I(latitude^3), data=forest_ants)

#' The `I()` function ensures that `^` is not interpreted as model formula
#' syntax. See `?formula` for more details about model formulae.

#' An even more convenient way uses the function `poly()`, which creates a
#' matrix of the polynomial terms.

poly(forest_ants$latitude, degree=3, raw=TRUE)

#' We can use this directly within a model formula

lm(richness ~ poly(latitude, degree=3, raw=TRUE), data=forest_ants)

#' A potential problem with polynomial models is that the higher order terms can
#' become almost perfectly correlated with one another, leading to models where
#' the parameters can't all be uniquely estimated. For example, for these data
#' the fourth order polynomial can be trained but for the fifth order polynomial
#' we can't determine a unique value for the highest order parameter, and the
#' parameter estimates remain the same as the fourth order model. We have
#' essentially run out of uniqueness among the polynomial terms due to the high
#' correlations.

lm(richness ~ poly(latitude, degree=4, raw=TRUE), data=forest_ants)
lm(richness ~ poly(latitude, degree=5, raw=TRUE), data=forest_ants)
cor(poly(forest_ants$latitude, degree=5, raw=TRUE))

#' This problem can be markedly reduced by using orthogonal polynomials, which
#' remove the correlation among the polynomial terms. Orthogonal polynomials are
#' the default type for `poly()`.

lm(richness ~ poly(latitude, degree=5), data=forest_ants)
cor(poly(forest_ants$latitude, degree=5))

#' Orthogonal polynomials give the same predictions as the raw polynomials. It's
#' just a difference in parameterization of the same model. In machine learning
#' we don't care about the parameter values, just the resulting prediction, so
#' it's best to choose the more robust parameterization.
#' 

#' R's `lm()` function contains a **training algorithm** that finds the
#' parameters that minimize the sum of squared deviations of the data from the
#' model. The following code trains the order 4 polynomial and plots the fitted
#' model. Use this block of code to try different values for the order of the
#' polynomial. We can get up to order 16, after which we can no longer form
#' orthogonal polynomials.

order <- 4 #integer
poly_trained <- lm(richness ~ poly(latitude, order), data=forest_ants)
grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
nd <- data.frame(latitude=grid_latitude)
pred_richness <- predict(poly_trained, newdata=nd)
preds <- cbind(nd, richness=pred_richness)

ggplot(data=NULL, aes(x=latitude, y=richness)) +
    geom_point(data=forest_ants) +
    geom_line(data=preds) +
    coord_cartesian(ylim=c(0,20)) +
    labs(title=paste("Polynomial order", order))


#' Use `predict` to ask for predictions from the trained polynomial model. For
#' example, here we are asking for the prediction at latitude 43.2 and we find
#' the predicted richness is 5.45. We need to provide the predictor variable
#' `latitude` as a data frame even if it's just one value. See `?predict.lm`.

predict(poly_trained, newdata=data.frame(latitude=43.2))

#' Exploring the k-fold CV algorithm
#'
#' First, we need a function to divide the dataset up into partitions.

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

#' What does the output of `random_partitions()` look like? It's a set of labels
#' that says which partition each data point belongs to.

random_partitions(nrow(forest_ants), k=5)
random_partitions(nrow(forest_ants), k=nrow(forest_ants))


#' Now code up the k-fold CV algorithm (from our pseudocode to R code) to
#' estimate the prediction mean squared error for one order of the polynomial.
#' Try 5-fold, 10-fold, and n-fold CV. Try different values for polynomial
#' order.

order <- 2
k <- 5

# divide dataset into k parts i = 1...k
forest_ants$partition <- random_partitions(nrow(forest_ants), k)

# initiate vector to hold mean squared errors
e <- rep(NA, k)

# for each i
for ( i in 1:k ) {
#   test dataset = part i
    test_data <- forest_ants |> filter(partition == i)
#   training dataset = remaining data
    train_data <- forest_ants |> filter(partition != i)
#   find f using training dataset
    poly_trained <- lm(richness ~ poly(latitude, order), data=train_data)
#   use f to predict for test dataset
    pred_richness <- predict(poly_trained, newdata=test_data)
#   e_i = prediction error (MSE)
    e[i] <- mean((test_data$richness - pred_richness)^2)
}
# CV_error = mean(e)
cv_error <- mean(e)
cv_error


#' To help us do systematic experiments to explore different combinations
#' of `order` and `k` we'll encapsulate the above code as a function.

# Function to perform k-fold CV for a polynomial model on ants data
# forest_ants: dataframe
# k:           number of partitions (scalar, integer)
# order:       degrees of polynomial (scalar, integer)
# return:      CV error as MSE (scalar, numeric)

cv_poly_ants <- function(forest_ants, k, order) {
    forest_ants$partition <- random_partitions(nrow(forest_ants), k)
    e <- rep(NA, k)
    for ( i in 1:k ) {
        test_data <- forest_ants |> filter(partition == i)
        train_data <- forest_ants |> filter(partition != i)
        poly_trained <- lm(richness ~ poly(latitude, order), data=train_data)
        pred_richness <- predict(poly_trained, newdata=test_data)
        e[i] <- mean((test_data$richness - pred_richness) ^ 2)
    }
    cv_error <- mean(e)
    return(cv_error)
}


#' Test the function
cv_poly_ants(forest_ants, k=10, order=4)
cv_poly_ants(forest_ants, k=22, order=4)


#' Explore a grid of values for k and polynomial order.
#' 
#' We could use nested iteration structures like this to calculate the CV error
#' for different combinations of k and order.

output <- matrix(nrow=24, ncol=3)
colnames(output) <- c("k", "order", "cv_error")
i <- 1
for ( k in c(5, 10, 22 ) ) {
    for (order in 1:8 ) {
        output[i,1:2] <- c(k, order)
        output[i,3] <- cv_poly_ants(forest_ants, k, order)
        i <- i + 1
    }
}
output

#' But a neater and easier solution uses the `expand.grid()` function. We'll
#' also set a random seed so that the result is repeatable.

set.seed(1193) #For reproducible results

grid <- expand.grid(k=c(5,10,nrow(forest_ants)), order=1:8 )
cv_error <- rep(NA, nrow(grid))
for( i in 1:nrow(grid) ) {
    cv_error[i] <- cv_poly_ants(forest_ants, k=grid$k[i], order=grid$order[i])
}
result1 <- cbind(grid, cv_error)
result1

#' Plot

result1 |>
    ggplot() +
    geom_line(aes(x=order, y=cv_error, col=factor(k)))

#' We see that prediction error is very large for order > 7. We need to adjust
#' the y-axis limits to zoom in.

result1 |>
    ggplot() +
    geom_line(aes(x=order, y=cv_error, col=factor(k))) +
    ylim(10,25)

#' but now the y limits break the line segments that fall outside the limits. We
#' need to use `coord_cartesian()` to set the limits instead.

result1 |>
    ggplot() +
    geom_line(aes(x=order, y=cv_error, col=factor(k))) +
    coord_cartesian(ylim=c(10,25))

#' We see that MSE prediction error (cv_error) generally increases for order
#' greater than 2 or 3. We also see that cv_error estimates are variable for
#' k=10 and especially k=5. This is due to the randomness of partitioning a very
#' small dataset. If we repeat the above with a different seed, we'd get
#' different results for k=5 or k=10. LOOCV is deterministic for this model, so
#' it won't differ if we repeat it.
#'
#' LOOCV (k=22) identifies order=2 as the best performing model, whereas in this
#' particular run 10-fold and 5-fold CV identify order=3.
#'
#' This variability illustrates that we should be mindful that k-fold CV can be
#' noisy. What should we do here? Given the uncertainty in MSE estimates for k =
#' 5 or 10, we'd be best to use LOOCV as a default (generally a good strategy
#' for small datasets). But we could also try for a better estimate by repeated
#' k-fold runs. Let's explore the variability in 5-fold and 10-fold CV.

#+ results=FALSE, cache=TRUE
set.seed(1978) #For reproducible results
grid <- expand.grid(k=c(5,10), order=1:7)
reps <- 100
cv_error <- matrix(NA, nrow=nrow(grid), ncol=reps)
for ( j in 1:reps ) {
    for ( i in 1:nrow(grid) ) {
        cv_error[i,j] <- cv_poly_ants(forest_ants, grid$k[i], grid$order[i])
    }
    print(j) #monitor progress
}
result2 <- cbind(grid, cv_error)

#' Plot the first 10 reps for each k-fold

result2 |> 
    select(1:12) |>
    mutate(k=paste(k, "-fold CV", sep="")) |>
    pivot_longer(cols="1":"10", names_to="rep", values_to="cv_error") |> 
    mutate(rep=as.numeric(rep)) |> 
    ggplot() +
    geom_line(aes(x=order, y=cv_error, col=factor(rep))) +
    facet_wrap(vars(k)) +
    coord_cartesian(ylim=c(10,25))

#' We see again that there is more variability for 5-fold CV. For both 5-fold
#' and 10-fold CV there is so much variability, we'd pick different values for
#' order on different runs. So, we wouldn't want to rely on a single k-fold run.
#' 
#' Averaging across runs would give a better estimate of the prediction MSE:

result2$mean_cv <- rowMeans(result2[,-(1:2)])

#' From the plot of the average for k = 5 and 10, we'd pick the same order as
#' LOOCV (k=22).

loocv <- result1 |> 
    filter(k == 22, order <= 7)

result2 |>
    select(k, order, mean_cv) |>
    rename(cv_error=mean_cv) |>
    bind_rows(loocv) |>
    ggplot() +
    geom_line(aes(x=order, y=cv_error, col=factor(k))) +
    labs(title=paste("Mean across",reps,"k-fold CV runs"), col="k") +
    coord_cartesian(ylim=c(10,25))

#' Finally, here is the table of results

result2 |>
    select(k, order, mean_cv) |>
    rename(cv_error=mean_cv) |>
    bind_rows(loocv) |>
    arrange(k)
