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


#' Here is one way we could code a 3rd order polynomial using R's model formula
#' syntax.

lm(richness ~ latitude + I(latitude^2) + I(latitude^3), data=forest_ants)

#' The `I()` function ensures that `^` is not interpreted as model formula
#' syntax. See `?formula` for more details. Briefly, model formulae provide a
#' shorthand notation for (mostly) linear models, e.g. `y ~ x + z` is
#' shorthand for the model:
#' 
#' $$
#' y = \beta_0 + \beta_1 x + \beta_2 z
#' $$
#' 

#' A more convenient way is the function `poly()`

lm(richness ~ poly(latitude, degree=3), data=forest_ants)

#' The difference in the parameter estimates from the previous approach is
#' because the parameterization is different. Use the argument `raw=TRUE` for
#' the same parameterization (see `?poly`). In machine learning we don't care
#' about the parameter values, just the resulting prediction, which is exactly
#' the same for the two approaches. R's `lm()` function contains a **training**
#' **algorithm** that finds the parameters that minimize the sum of squared
#' deviations of the data from the model.
#' 

#' Example plot of an order 4 polynomial model. Use this block of code to try
#' different values for the order (syn. degree) of the polynomial.

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

