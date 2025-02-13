Ant data: random forest
================
Brett Melbourne
15 Feb 2024

Random forest illustrated with the ants data.

``` r
library(ggplot2)
library(dplyr)
library(tree)
library(randomForest)
source("source/bagrt.R")
```

Ant data with 3 predictors of species richness

``` r
ants <- read.csv("data/ants.csv") |> 
    select(richness, latitude, habitat, elevation) |> 
    mutate(habitat=factor(habitat))
```

## Hand coded random forest algorithm

**Random forest** is an ensemble prediction method that extends the
bagging algorithm by randomly selecting a subset of predictor variables
at each iteration. That is, random forests combines two strategies for
enhancing generalization: resampling the data (rows) and resampling the
predictors (columns).

The random forest algorithm:

    for many repetitions
        randomly select m predictor variables
        resample the data (rows) with replacement
        train the tree model
        record prediction
    final prediction = mean of predictions

Code this algorithm in R

Since this code is a proof of concept, to keep the algorithm clear and
uncluttered by book-keeping, we’ll require that the dataset be given
with the response variable (`richness`) in the first column and the full
set of predictors in the other columns. We did that when reading in the
data above.

``` r
# Random forest algorithm

# Grid of new predictor values on which to form predictions
grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=51))

# Parameters
m <- 2 #Number of predictors to sample at each iteration
boot_reps <- 500

# Setup
n <- nrow(ants)
c <- ncol(ants)
nn <- nrow(grid_data)
boot_preds <- matrix(rep(NA, nn*boot_reps), nrow=nn, ncol=boot_reps)

# Main algorithm
for ( i in 1:boot_reps ) {
#   randomly select m predictor variables
    predictor_indices <- sample(2:c, m)
    boot_data <- ants[,c(1,predictor_indices)]
#   resample the data (rows) with replacement
    boot_indices <- sample(1:n, n, replace=TRUE)
    boot_data <- boot_data[boot_indices,]
#   train the tree model
    boot_fit <- tree(richness ~ ., data=boot_data)
#   record prediction
    boot_preds[,i] <- predict(boot_fit, newdata=grid_data)
}
rf_preds <- rowMeans(boot_preds)
```

Plot predictions with elevation mapped to the viridis color scale (this
color scale has [nice
properties](https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html)).
It’s available directly from `ggplot()` without loading an extra
package. The `c` in `scale_color_viridis_c()` stands for “continuous
variable”.

``` r
preds <- cbind(grid_data, richness=rf_preds)
ants |> 
    ggplot() +
    geom_line(data=preds, 
              aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(aes(x=latitude, y=richness, col=elevation)) +
    facet_wrap(vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()
```

![](05_5_ants_random_forest_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Before moving on, we’ll make the algorithm into a function. We’re not
especially concerned with performance or elegance. One small performance
boost we can easily make is reduce two copy operations to one by
extracting columns and rows at the same time. On the other hand, while
we could automatically extract the name of the response variable and
build a model formula we’ll leave it up to the user to provide a correct
formula and dataset (it’s only for our use and documented in the
function). This will also be consistent with our previous functions.

``` r
# Random forest function
# formula:    model formula to indicate response variable (formula)
#             must be: y ~ ., where y is the name of the variable
# data:       y and x data (data.frame, mixed)
#             y must be first column; all other columns will be used for x
# xnew_data:  x data to predict from (data.frame, mixed)
# m:          number of predictors to sample (scalar, integer)
# boot_reps:  number of bootstrap replications (scalar, integer)
# return:     bagged predictions (vector, numeric)
# 
random_forest <- function(formula, data, xnew_data, m, boot_reps=500) {
    # Setup
    n <- nrow(data)
    c <- ncol(data)
    nn <- nrow(xnew_data)
    boot_preds <- matrix(rep(NA, nn*boot_reps), nrow=nn, ncol=boot_reps)
    
    # Main algorithm
    for ( i in 1:boot_reps ) {
    #   randomly select m predictor variables
        predictor_indices <- sample(2:c, m)
    #   resample the data (rows) with replacement
        boot_indices <- sample(1:n, n, replace=TRUE)
    #   form the boot dataset
        boot_data <- data[boot_indices, c(1,predictor_indices)]
    #   train the tree model
        boot_fit <- tree(formula, data=boot_data)
    #   record prediction
        boot_preds[,i] <- predict(boot_fit, newdata=xnew_data)
    }
    rf_preds <- rowMeans(boot_preds)
    return(rf_preds)
}
```

Here’s how to call it. Check that it works using plotting code above to
confirm the plot is the same. The notation `~ .` means use all the
predictor variables in the data frame.

``` r
preds_rf <- random_forest(richness ~ ., data=ants, xnew_data=grid_data, m=2)
```

Train a bagged tree and a single decision tree for comparison

``` r
preds_bag <- bagrt(richness ~ ., data=ants, xnew_data=grid_data)
preds_tree <- predict(tree(richness ~ ., data=ants), newdata=grid_data)
```

Plot together

``` r
preds_rf <- data.frame(grid_data, richness=preds_rf, model="rf")
preds_bag <- data.frame(grid_data, richness=preds_bag, model="bag")
preds_tree <- data.frame(grid_data, richness=preds_tree, model="tree")

rbind(preds_rf, preds_bag, preds_tree) |>
    ggplot() +
    geom_line(aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(data=ants, aes(x=latitude, y=richness, col=elevation)) +
    facet_grid(rows=vars(model), cols=vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()
```

![](05_5_ants_random_forest_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

## The randomForest package

Using `randomForest()` from the randomForest package. Compared with our
proof of concept code, the difference is due to the implementation and
default settings of the base tree algorithm.

``` r
rf_pkg_train <- randomForest(richness ~ ., data=ants, ntree=500, mtry=2)
preds_rf_pkg <- predict(rf_pkg_train, newdata=grid_data)
preds_rf_pkg <- data.frame(grid_data, richness=preds_rf_pkg, model="rf_pkg")

rbind(preds_rf, preds_rf_pkg) |>
    ggplot() +
    geom_line(aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(data=ants, aes(x=latitude, y=richness, col=elevation)) +
    facet_grid(rows=vars(model), cols=vars(habitat)) +
    scale_color_viridis_c() +
    labs(title="Top row: proof of concept code. Bottom row: randomForest package") +
    theme_bw()
```

![](05_5_ants_random_forest_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

Tuning the random forest algorithm. We can use out-of-bag error for
number of trees and mtry.

``` r
# Visualize OOB error for number of trees
plot(rf_pkg_train)
```

![](05_5_ants_random_forest_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
# OOB error for mtry
tuneRF(x=ants[,-1], y=ants[,1], mtryStart=3, ntreeTry=500)
```

    ## mtry = 3  OOB error = 11.70326 
    ## Searching left ...
    ## mtry = 2     OOB error = 11.45618 
    ## 0.02111195 0.05 
    ## Searching right ...

![](05_5_ants_random_forest_files/figure-gfm/unnamed-chunk-10-2.png)<!-- -->

    ##   mtry OOBError
    ## 2    2 11.45618
    ## 3    3 11.70326

For `tuneRF()` with OOB samples, it is of course stochastic and could
vary a lot from run to run on small datasets like `ants`, just as we’ve
seen such tuning stochasticity with k-fold CV. Try running the above
line several times. Use k-fold CV if necessary and with repeated random
splits if necessary. Here, because of the stochasticity in the ants
dataset, we’d want to use regular CV with repeated random splits as the
OOB error from tuneRF is too noisy. Why are about 1/3 out of bag?
Because we are sampling n with replacement. Here’s a quick simulation
sampling with replacement from n=100 rows.

``` r
reps <- 100000
n_in_bag <- rep(NA, reps)
for ( i in 1:reps ) {
    inbag <- sample(1:100, size=100, replace=TRUE)
    inbag <- unique(inbag)
    n_in_bag[i] <- sum(1:100 %in% inbag)
}
OOB <- 100 - mean(n_in_bag)
OOB #36.6% are out of bag
```

    ## [1] 36.6068

In general, we want to tune number of trees, mtry, and the minimum node
size of trees (a tree stopping rule). You would proceed as in
`ants_bag.R`.

## Explainable machine learning

Variable importance

``` r
importance(rf_pkg_train)
```

    ##           IncNodePurity
    ## latitude       312.0733
    ## habitat        183.0783
    ## elevation      201.2700

``` r
varImpPlot(rf_pkg_train)
```

![](05_5_ants_random_forest_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

## Bagging is a special case of random forest

Finally, bagged decision trees are a special case of the random forest
algorithm where the predictors are not randomly selected. So, in
general, we can use a random forest algorithm to do bagged regression
and classification trees. This produces similar results to our earlier
bagging code (differences are due to differences in the base decision
tree algorithm). To do bagging, we set `mtry` equal to the number of
predictors. We would want to tune the `nodesize` parameter, which I have
not done here. Here is a random forest compared to a bagged regression
tree, both using `randomForest()`, for two predictors.

``` r
# Bagged tree with latitude and habitat as predictors
bag_train <- randomForest(richness ~ latitude + habitat, 
                         data=ants, ntree=500, mtry=2, nodesize=10)

# Random forest with latitude and habitat as predictors
forest_train <- randomForest(richness ~ latitude + habitat, 
                         data=ants, ntree=500, mtry=1, nodesize=10)

# Predictions for both models
grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")))
bag_pred <- predict(bag_train, newdata=grid_data)
forest_pred <- predict(forest_train, newdata=grid_data)

preds_bag <- data.frame(grid_data, richness=bag_pred, model="bag (mtry=2)")
preds_forest <- data.frame(grid_data, richness=forest_pred, model="rf (mtry=1)")

rbind(preds_bag, preds_forest) |>
    ggplot() +
    geom_line(aes(x=latitude, y=richness, col=habitat)) +
    geom_point(data=ants, aes(x=latitude, y=richness, col=habitat)) +
    facet_grid(cols=vars(model)) +
    labs(title="Random forest compared to bagged regression tree") +
    theme_bw()
```

![](05_5_ants_random_forest_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->
