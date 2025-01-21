Ant data: k-fold cross validation
================
Brett Melbourne
21 Jan 2025

Investigate cross-validation **inference algorithm** with ants data and
a polynomial model. Our goal is to predict richness of forest ants from
latitude. What order of a polynomial **model algorithm** gives the most
accurate predictions?

``` r
library(ggplot2)
library(dplyr)
library(tidyr) #for pivot_longer()
```

Ant data:

``` r
ants <- read.csv("data/ants.csv")
head(ants)
```

    ##   site habitat latitude elevation richness
    ## 1  TPB  forest    41.97       389        6
    ## 2  HBC  forest    42.00         8       16
    ## 3  CKB  forest    42.03       152       18
    ## 4  SKP  forest    42.05         1       17
    ## 5   CB  forest    42.05       210        9
    ## 6   RP  forest    42.17        78       15

Forest ant data:

``` r
forest_ants <- ants |> 
    filter(habitat=="forest")

forest_ants |>
    ggplot() +
    geom_point(aes(x=latitude, y=richness)) +
    ylim(0,20)
```

![](02_2_ants_cv_polynomial_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

Here is one way we could code a 3rd order polynomial using R’s model
formula syntax.

``` r
lm(richness ~ latitude + I(latitude^2) + I(latitude^3), data=forest_ants)
```

    ## 
    ## Call:
    ## lm(formula = richness ~ latitude + I(latitude^2) + I(latitude^3), 
    ##     data = forest_ants)
    ## 
    ## Coefficients:
    ##   (Intercept)       latitude  I(latitude^2)  I(latitude^3)  
    ##    84336.3595     -5736.2100       130.0406        -0.9825

The `I()` function ensures that `^` is not interpreted as model formula
syntax. See `?formula` for more details. Briefly, model formulae provide
a shorthand notation for (mostly) linear models, e.g. `y ~ x + z` is
shorthand for the model:

$$
y = \beta_0 + \beta_1 x + \beta_2 z
$$

A more convenient way is the function `poly()`

``` r
lm(richness ~ poly(latitude, degree=3), data=forest_ants)
```

    ## 
    ## Call:
    ## lm(formula = richness ~ poly(latitude, degree = 3), data = forest_ants)
    ## 
    ## Coefficients:
    ##                 (Intercept)  poly(latitude, degree = 3)1  
    ##                       9.182                      -11.604  
    ## poly(latitude, degree = 3)2  poly(latitude, degree = 3)3  
    ##                       6.170                       -2.629

The difference in the parameter estimates from the previous approach is
because the parameterization is different. Use the argument `raw=TRUE`
for the same parameterization (see `?poly`). In machine learning we
don’t care about the parameter values, just the resulting prediction,
which is exactly the same for the two approaches. R’s `lm()` function
contains a **training** **algorithm** that finds the parameters that
minimize the sum of squared deviations of the data from the model.

Example plot of an order 4 polynomial model. Use this block of code to
try different values for the order (syn. degree) of the polynomial.

``` r
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
```

![](02_2_ants_cv_polynomial_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Using `predict` to ask for predictions from the trained polynomial
model. For example, here we are asking for the prediction at latitude
43.2 and we find the predicted richness is 5.45.

``` r
predict(poly_trained, newdata=data.frame(latitude=43.2))
```

    ##        1 
    ## 5.450208

Exploring the k-fold CV algorithm

First, we need a function to divide the dataset up into partitions.

``` r
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
```

What does the output of `random_partitions()` look like? It’s a set of
labels that says which partition each data point belongs to.

``` r
random_partitions(nrow(forest_ants), k=5)
```

    ##  [1] 2 1 5 2 1 1 5 1 2 5 4 5 4 2 3 2 4 3 3 4 3 1

``` r
random_partitions(nrow(forest_ants), k=nrow(forest_ants))
```

    ##  [1] 21 15  4  1 12 10  3  5 16  6  2 13 14  8  9 17 11  7 18 22 19 20

Now code up the k-fold CV algorithm to estimate the prediction mean
squared error for one order of the polynomial. Try 5-fold, 10-fold, and
n-fold CV. Try different values for polynomial order.

``` r
# divide dataset into k parts i = 1...k
# for each i
#     test dataset = part i
#     training dataset = remaining data
#     find f using training dataset
#     use f to predict for test dataset 
#     e_i = prediction error
# CV_error = mean(e)
```
