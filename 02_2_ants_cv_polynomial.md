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

Here’s one way we could code a 3rd order polynomial by first creating
new variables for the squared (quadratic) and cubed (cubic) terms, and
using R’s model formula syntax to train the model by minimizing the SSQ
with the function `lm`.

``` r
forest_ants$latitude_2 <- forest_ants$latitude ^ 2
forest_ants$latitude_3 <- forest_ants$latitude ^ 3
head(forest_ants)
```

    ##   site habitat latitude elevation richness latitude_2 latitude_3
    ## 1  TPB  forest    41.97       389        6   1761.481   73929.35
    ## 2  HBC  forest    42.00         8       16   1764.000   74088.00
    ## 3  CKB  forest    42.03       152       18   1766.521   74246.87
    ## 4  SKP  forest    42.05         1       17   1768.202   74352.92
    ## 5   CB  forest    42.05       210        9   1768.202   74352.92
    ## 6   RP  forest    42.17        78       15   1778.309   74991.29

``` r
lm(richness ~ latitude + latitude_2 + latitude_3, data=forest_ants)
```

    ## 
    ## Call:
    ## lm(formula = richness ~ latitude + latitude_2 + latitude_3, data = forest_ants)
    ## 
    ## Coefficients:
    ## (Intercept)     latitude   latitude_2   latitude_3  
    ##  84336.3595   -5736.2100     130.0406      -0.9825

A model formula provides a shorthand notation for (mostly) linear
models, e.g. `y ~ x + z` is shorthand for the model:

$$
y = \beta_0 + \beta_1 x + \beta_2 z
$$

Here’s another way to code the same model that eliminates the need to
create new variables for higher order terms.

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
syntax. See `?formula` for more details about model formulae. An even
more convenient way uses the function `poly()`, which creates a matrix
of the polynomial terms.

``` r
poly(forest_ants$latitude, degree=3, raw=TRUE)
```

    ##           1        2        3
    ##  [1,] 41.97 1761.481 73929.35
    ##  [2,] 42.00 1764.000 74088.00
    ##  [3,] 42.03 1766.521 74246.87
    ##  [4,] 42.05 1768.202 74352.92
    ##  [5,] 42.05 1768.202 74352.92
    ##  [6,] 42.17 1778.309 74991.29
    ##  [7,] 42.19 1779.996 75098.04
    ##  [8,] 42.23 1783.373 75311.84
    ##  [9,] 42.27 1786.753 75526.05
    ## [10,] 42.31 1790.136 75740.66
    ## [11,] 42.56 1811.354 77091.21
    ## [12,] 42.57 1812.205 77145.56
    ## [13,] 42.58 1813.056 77199.94
    ## [14,] 42.69 1822.436 77799.80
    ## [15,] 43.33 1877.489 81351.59
    ## [16,] 44.06 1941.284 85532.96
    ## [17,] 44.29 1961.604 86879.45
    ## [18,] 44.33 1965.149 87115.05
    ## [19,] 44.50 1980.250 88121.12
    ## [20,] 44.55 1984.702 88418.50
    ## [21,] 44.76 2003.458 89674.76
    ## [22,] 44.95 2020.503 90821.59
    ## attr(,"degree")
    ## [1] 1 2 3
    ## attr(,"class")
    ## [1] "poly"   "matrix"

We can use this directly within a model formula

``` r
lm(richness ~ poly(latitude, degree=3, raw=TRUE), data=forest_ants)
```

    ## 
    ## Call:
    ## lm(formula = richness ~ poly(latitude, degree = 3, raw = TRUE), 
    ##     data = forest_ants)
    ## 
    ## Coefficients:
    ##                             (Intercept)  
    ##                              84336.3595  
    ## poly(latitude, degree = 3, raw = TRUE)1  
    ##                              -5736.2100  
    ## poly(latitude, degree = 3, raw = TRUE)2  
    ##                                130.0406  
    ## poly(latitude, degree = 3, raw = TRUE)3  
    ##                                 -0.9825

A potential problem with polynomial models is that the higher order
terms can become almost perfectly correlated with one another, leading
to models where the parameters can’t all be uniquely estimated. For
example, for these data the fourth order polynomial can be trained but
the fifth order polynomial can’t determine a unique value for the
highest order parameter, and the parameter estimates remain the same as
the fourth order model.

``` r
lm(richness ~ poly(latitude, degree=4, raw=TRUE), data=forest_ants)
```

    ## 
    ## Call:
    ## lm(formula = richness ~ poly(latitude, degree = 4, raw = TRUE), 
    ##     data = forest_ants)
    ## 
    ## Coefficients:
    ##                             (Intercept)  
    ##                              -1.615e+06  
    ## poly(latitude, degree = 4, raw = TRUE)1  
    ##                               1.509e+05  
    ## poly(latitude, degree = 4, raw = TRUE)2  
    ##                              -5.281e+03  
    ## poly(latitude, degree = 4, raw = TRUE)3  
    ##                               8.209e+01  
    ## poly(latitude, degree = 4, raw = TRUE)4  
    ##                              -4.781e-01

``` r
lm(richness ~ poly(latitude, degree=5, raw=TRUE), data=forest_ants)
```

    ## 
    ## Call:
    ## lm(formula = richness ~ poly(latitude, degree = 5, raw = TRUE), 
    ##     data = forest_ants)
    ## 
    ## Coefficients:
    ##                             (Intercept)  
    ##                              -1.615e+06  
    ## poly(latitude, degree = 5, raw = TRUE)1  
    ##                               1.509e+05  
    ## poly(latitude, degree = 5, raw = TRUE)2  
    ##                              -5.281e+03  
    ## poly(latitude, degree = 5, raw = TRUE)3  
    ##                               8.209e+01  
    ## poly(latitude, degree = 5, raw = TRUE)4  
    ##                              -4.781e-01  
    ## poly(latitude, degree = 5, raw = TRUE)5  
    ##                                      NA

``` r
cor(poly(forest_ants$latitude, degree=5, raw=TRUE))
```

    ##           1         2         3         4         5
    ## 1 1.0000000 0.9999785 0.9999141 0.9998067 0.9996564
    ## 2 0.9999785 1.0000000 0.9999785 0.9999141 0.9998067
    ## 3 0.9999141 0.9999785 1.0000000 0.9999785 0.9999141
    ## 4 0.9998067 0.9999141 0.9999785 1.0000000 0.9999785
    ## 5 0.9996564 0.9998067 0.9999141 0.9999785 1.0000000

This problem can be markedly reduced by using orthogonal polynomials,
which reduce the correlation among the polynomial terms. Orthogonal
polynomials are the default type for `poly()`.

``` r
lm(richness ~ poly(latitude, degree=5), data=forest_ants)
```

    ## 
    ## Call:
    ## lm(formula = richness ~ poly(latitude, degree = 5), data = forest_ants)
    ## 
    ## Coefficients:
    ##                 (Intercept)  poly(latitude, degree = 5)1  
    ##                      9.1818                     -11.6039  
    ## poly(latitude, degree = 5)2  poly(latitude, degree = 5)3  
    ##                      6.1704                      -2.6291  
    ## poly(latitude, degree = 5)4  poly(latitude, degree = 5)5  
    ##                     -0.8511                       2.7290

``` r
cor(poly(forest_ants$latitude, degree=5))
```

    ##               1             2            3             4             5
    ## 1  1.000000e+00 -4.567202e-18 9.958397e-17 -1.168363e-16  2.687466e-17
    ## 2 -4.567202e-18  1.000000e+00 1.086913e-17  1.630369e-17 -5.149960e-19
    ## 3  9.958397e-17  1.086913e-17 1.000000e+00  4.305638e-17  6.166400e-18
    ## 4 -1.168363e-16  1.630369e-17 4.305638e-17  1.000000e+00 -1.918360e-17
    ## 5  2.687466e-17 -5.149960e-19 6.166400e-18 -1.918360e-17  1.000000e+00

Orthogonal polynomials give the same predictions as the raw polynomials.
It is just a difference in parameterization of the same model. In
machine learning we don’t care about the parameter values, just the
resulting prediction, so it’s best to choose the more robust
parameterization. R’s `lm()` function contains a **training algorithm**
that finds the parameters that minimize the sum of squared deviations of
the data from the model.

Example plot of an order 4 polynomial model. Use this block of code to
try different values for the order (syn. degree) of the polynomial. We
can get up to order 16, after which we can no longer form orthogonal
polynomials.

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

![](02_2_ants_cv_polynomial_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

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

    ##  [1] 4 5 4 2 1 1 2 1 2 2 3 3 5 4 4 1 1 3 5 2 5 3

``` r
random_partitions(nrow(forest_ants), k=nrow(forest_ants))
```

    ##  [1] 17  9  7 13  1 22 18  6  5 20 12 14 11 16 10 15  4 19 21  2  8  3

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
