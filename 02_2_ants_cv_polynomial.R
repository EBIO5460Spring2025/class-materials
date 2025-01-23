#' ---
#' title: "Ant data: k-fold cross validation"
#' author: Brett Melbourne
#' date: 21 Jan 2025
#' output:
#'     github_document
#' ---

#' Investigate cross-validation **inference algorithm** with ants data and a
#' polynomial model. Our goal is to predict richness of forest ants from
#' latitude. What order of a polynomial **model algorithm** gives the most
#' accurate predictions?

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
#' variables for the squared (quadratic) and cubed (cubic) terms, and using R's
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
#' the fourth order polynomial can be trained but the fifth order polynomial
#' can't determine a unique value for the highest order parameter, and the
#' parameter estimates remain the same as the fourth order model.

lm(richness ~ poly(latitude, degree=4, raw=TRUE), data=forest_ants)
lm(richness ~ poly(latitude, degree=5, raw=TRUE), data=forest_ants)
cor(poly(forest_ants$latitude, degree=5, raw=TRUE))

#' This problem can be markedly reduced by using orthogonal polynomials, which
#' reduce the correlation among the polynomial terms. Orthogonal polynomials are
#' the default type for `poly()`.

lm(richness ~ poly(latitude, degree=5), data=forest_ants)
cor(poly(forest_ants$latitude, degree=5))

#' Orthogonal polynomials give the same predictions as the raw polynomials. It
#' is just a difference in parameterization of the same model. In machine
#' learning we don't care about the parameter values, just the resulting
#' prediction, so it's best to choose the more robust parameterization. R's
#' `lm()` function contains a **training algorithm** that finds the parameters
#' that minimize the sum of squared deviations of the data from the model.
#' 

#' Example plot of an order 4 polynomial model. Use this block of code to try
#' different values for the order (syn. degree) of the polynomial. We can get up
#' to order 16, after which we can no longer form orthogonal polynomials.

order <- 4 #integer
poly_trained <- lm(richness ~ poly(latitude, order), data=forest_ants)
grid_latitude  <- seq(min(forest_ants$latitude), max(forest_ants$latitude), length.out=201)
nd <- data.frame(latitude=grid_latitude)
pred_richness <- predict(poly_trained, newdata=nd)
preds <- cbind(nd, richness=pred_richness)

forest_ants |>
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    geom_line(data=preds, aes(x=latitude, y=richness)) +
    coord_cartesian(ylim=c(0,20)) +
    labs(title=paste("Polynomial order", order))


#' Using `predict` to ask for predictions from the trained polynomial model. For
#' example, here we are asking for the prediction at latitude 43.2 and we find
#' the predicted richness is 5.45.

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


#' Now code up the k-fold CV algorithm to estimate the prediction mean squared
#' error for one order of the polynomial. Try 5-fold, 10-fold, and n-fold CV.
#' Try different values for polynomial order.

# divide dataset into k parts i = 1...k
# for each i
#     test dataset = part i
#     training dataset = remaining data
#     find f using training dataset
#     use f to predict for test dataset 
#     e_i = prediction error
# CV_error = mean(e)

