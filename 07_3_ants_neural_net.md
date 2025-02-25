Ant data: neural network
================
Brett Melbourne
27 Feb 2024

A single layer neural network, or feedforward network, illustrated with
the ants data. We first hand code the model algorithm as a proof of
understanding. Then we code the same model and train it using Keras.

``` r
library(ggplot2)
library(dplyr)
```

Ant data with 3 predictors of species richness

``` r
ants <- read.csv("data/ants.csv") |> 
    select(richness, latitude, habitat, elevation)
head(ants)
```

    ##   richness latitude habitat elevation
    ## 1        6    41.97  forest       389
    ## 2       16    42.00  forest         8
    ## 3       18    42.03  forest       152
    ## 4       17    42.05  forest         1
    ## 5        9    42.05  forest       210
    ## 6       15    42.17  forest        78

For neural networks it is customary to scale numeric predictors to zero
mean, unit variance, as the training algorithms perform better. To make
predictions with new data, we need to use the scaling parameters from
the data used to train the model. Here, we calculate the scaling
parameters (mean and standard deviation) to use later.

``` r
lat_mn <- mean(ants$latitude)
lat_sd <- sd(ants$latitude)
ele_mn <- mean(ants$elevation)
ele_sd <- sd(ants$elevation)
```

### Hand-coded feedforward network

Before we go on to Keras, we’re going to hand code a feedforward network
in R to gain a good understanding of the model algorithm. Here is our
pseudocode from the lecture notes.

    Single layer neural network, model algorithm:

    define g(z)
    load X, i=1...n, j=1...p (in appropriate form)
    set K
    set weights and biases: w(1)_jk, b(1)_k, w(2)_k1, b(2)_1
    for each activation unit k in 1:K
        calculate linear predictor: z_k = b(1)_k + Xw(1)_k
        calculate nonlinear activation: A_k = g(z_k)
    calculate linear model: f(X) = b(2)_1 + Aw(2)_1
    return f(X)

Now code this algorithm in R. I already trained this model using Keras
(see later) to obtain a parameter set for the weights and biases.

``` r
# Single layer neural network, model algorithm

# define g(z)
g_relu <- function(z) {
    g_z <- ifelse(z < 0, 0, z)
    return(g_z)
}

# load x (could be a grid of new predictor values or the original data)
grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=c("forest","bog"),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=51))

# data preparation: scale, one-hot encoding, convert to matrix
x <- grid_data |>
    mutate(latitude = (latitude - lat_mn) / lat_sd,
           elevation = (elevation - ele_mn) / ele_sd,
           bog = ifelse(habitat == "bog", 1, 0),
           forest = ifelse(habitat == "forest", 1, 0)) |>    
    select(latitude, bog, forest, elevation) |>     #drop richness & habitat
    as.matrix()

# dimensions of x
n <- nrow(x)
p <- ncol(x)

# set K
K <- 5

# set parameters (weights and biases)
w1 <- c(-0.2514450848, 0.4609818,  0.1607399, -0.9136779, -1.0339828,
        -0.4243144095, 0.7681985, -0.1529205, -0.3439012,  0.8026423,
        -0.0005548226, 0.6407318,  1.4387618,  1.6372939,  1.1695395,
        -0.0395327508, 0.5222837, -0.6239772, -0.3365386, -0.7156096) |>
    matrix(nrow=4, ncol=5, byrow=TRUE)

b1 <- c(-0.2956778, 0.3149067, 0.8800480, 0.6910487, 0.6947369)

w2 <- c(-0.4076283,
         0.6379358,
         0.8768858,
         1.6320601,
         0.9864114) |> 
    matrix(nrow=5, ncol=1, byrow=TRUE)

b2 <- 0.8487959


# hidden layer 1, iterating over each activation unit
A <- matrix(NA, nrow=n, ncol=K)
for ( k in 1:K ) {
#   linear predictor
    z <- x %*% w1[,k] + b1[k]
#   nonlinear activation
    A[,k] <- g_relu(z)
}

# output, layer 2, linear model
f_x <- A %*% w2  + b2

# return f(x); a redundant copy but mirrors our previous examples
nn1_preds <- f_x
```

Plot predictions

``` r
preds <- cbind(grid_data, richness=nn1_preds)
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

![](07_3_ants_neural_net_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### Using Keras to fit neural networks

Now we’ll consider the very same model with Keras, including training
the model on the data to determine the weight and bias parameters.

You’ll need to install Python and Tensorflow to use the `keras` package.
The `keras` package is an R interface to the Python Keras library, which
in turn is an interface to the Python Tensorflow library, which in turn
is an interface to Tensorflow (mostly C++)! See Assignment 4 for
installation directions. The Python Keras library is widely used and the
R functions and workflow closely mirror the Python functions and
workflow, so what we’ll learn in Keras for R largely applies to Keras
for Python as well. See also the Python version of this script.

``` r
# If you have another conda environment as your R default, you will need the
# following line before loading the keras library
reticulate::use_condaenv(condaenv = "r-tensorflow")
library(keras)
```

First we’ll set a random seed for reproducibility. The seed applies to
R, Python, and Tensorflow. It will take a few moments for Tensorflow to
get set up.

``` r
tensorflow::set_random_seed(5574)
```

Next, prepare the data (the predictors are prepared exactly as we did
for the hand-coded version above):

``` r
xtrain <- ants |> 
    mutate(latitude = (latitude - lat_mn) / lat_sd,
           elevation = (elevation - ele_mn) / ele_sd,
           bog = ifelse(habitat == "bog", 1, 0),
           forest = ifelse(habitat == "forest", 1, 0)) |>    
    select(latitude, bog, forest, elevation) |>     #drop richness & habitat
    as.matrix()

ytrain <- ants[,"richness"]
```

Next, specify the model. The basic syntax builds the model layer by
layer, using the pipe operator to express the flow of information from
the output of one layer to the input of the next. Here,
`keras_model_sequential()` says we will build a sequential (or
feedforward) network and the input layer will have `ncol(xtrain)` units
(i.e. there will be one input unit, or node, for each of the 4 predictor
columns in our ant data). The next layer, a hidden layer, will be a
densely-connected layer (i.e. all units from the previous layer
connected to all units of the hidden layer) with 5 units, and it will be
passed through the ReLU activation function. The output layer will be
another densely-connected layer with 1 unit and no activation applied.

``` r
modnn1 <- keras_model_sequential(input_shape = ncol(xtrain)) |>
    layer_dense(units = 5) |>
    layer_activation("relu") |> 
    layer_dense(units = 1)
```

We can check the configuration:

``` r
modnn1
```

    ## Model: "sequential"
    ## ________________________________________________________________________________
    ##  Layer (type)                       Output Shape                    Param #     
    ## ================================================================================
    ##  dense_1 (Dense)                    (None, 5)                       25          
    ##  activation (Activation)            (None, 5)                       0           
    ##  dense (Dense)                      (None, 1)                       6           
    ## ================================================================================
    ## Total params: 31 (124.00 Byte)
    ## Trainable params: 31 (124.00 Byte)
    ## Non-trainable params: 0 (0.00 Byte)
    ## ________________________________________________________________________________

The model and layer names, such as “dense_1” and “activation” are names
given to the objects created in the tensorflow workspace for this R
session. We see that layer 1 has 25 parameters as we expect (5x4 weights
and 5 biases) and the output layer has 6 parameters as we expect (5
weights and 1 bias). This model thus has a total of 31 parameters.

Next, compile the model. We are specifying that the optimizer algorithm
is RMSprop and the loss function is mean squared error. Notice that we
are not modifying the R `modnn1` object but are instead directing Keras
in Python to set up for training the model.

``` r
compile(modnn1, optimizer="rmsprop", loss="mse")
```

The RMSprop algorithm is the default in Keras and works for most models.
It implements stochastic gradient descent with an adaptive learning rate
and various performance enhancements. By default, the initial learning
rate is 0.001 (see `?optimizer_rmsprop`) but this can be tuned. Other
optimizers are available.

Now train the model, keeping a copy of the training history. Again, we
are not getting a new R fitted-model object out of this as we would in,
say, a call to `fit()` with an `lm` object but instead we are directing
Keras in Python to train the model (all the important output remains in
Python data structures). The history object is a by-product of training,
so I put it on the right-hand side of the expression to be clear that
this is a by-product and not a traditional R fitted-model object (where
you would usually expect to find the trained parameter values). In one
epoch, the training data are sampled in batches (one SGD step is taken
for each batch) until all of the data have been sampled, so the number
of epochs is the number of complete iterations through the training
data. I chose 300 epochs because the fit improves only slowly beyond
that. Here I used a batch size of 4, which means 10% of the data are
used on each subsample to calculate the stochastic gradient descent
step. Training will take a minute or so and a plot will chart its
progress.

``` r
fit(modnn1, xtrain, ytrain, epochs = 300, batch_size=4) -> history
```

As it takes time to train these models, it’s worth saving the model
(Tensorflow format) and history (R format) so they they can be reloaded
later. We can also load this saved model in Python or share with
colleagues.

``` r
# save_model_tf(modnn1, "07_3_ants_neural_net_files/saved/modnn1")
# save(history, file="07_3_ants_neural_net_files/saved/modnn1_history.Rdata")
modnn1 <- load_model_tf("07_3_ants_neural_net_files/saved/modnn1")
load("07_3_ants_neural_net_files/saved/modnn1_history.Rdata")
```

We can plot the history once training is done. If you have `ggplot2`
loaded, it will create a ggplot, otherwise it will create a base plot.

``` r
plot(history, smooth=FALSE, theme_bw=TRUE)
```

![](07_3_ants_neural_net_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

We want to see the training error (RMS) decline to a reasonable level.
Although the error will continue to go down, here we see it leveling out
somewhat at about RMS=7, which is an absolute error of sqrt(7) = +/- 2.6
species.

Make predictions for our grid of new predictor variables (`x`; we made
this earlier, scaling by the mean and sd of the data) and plot the
fitted model with the data. This plot is exactly the same as the one we
produced earlier “by hand” since they are the same model.

``` r
npred <- predict(modnn1, x)
```

    ## 641/641 - 1s - 932ms/epoch - 1ms/step

``` r
preds <- cbind(grid_data, richness=npred)
ants |> 
    ggplot() +
    geom_line(data=preds, 
              aes(x=latitude, y=richness, col=elevation,
                  group=factor(elevation)),
              linetype=2) +
    geom_point(aes(x=latitude, y=richness, col=elevation)) +
    facet_wrap(vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()
```

![](07_3_ants_neural_net_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

We can get the weights and biases, returned as a list

``` r
get_weights(modnn1)
```

    ## [[1]]
    ##               [,1]      [,2]       [,3]       [,4]       [,5]
    ## [1,] -0.2514450848 0.4609818  0.1607397 -0.9136779 -1.0339824
    ## [2,] -0.4243144095 0.7681987 -0.1529205 -0.3439011  0.8026420
    ## [3,] -0.0005548219 0.6407320  1.4387615  1.6372938  1.1695395
    ## [4,] -0.0395327471 0.5222836 -0.6239771 -0.3365385 -0.7156097
    ## 
    ## [[2]]
    ## [1] -0.2956778  0.3149068  0.8800483  0.6910489  0.6947370
    ## 
    ## [[3]]
    ##            [,1]
    ## [1,] -0.4076282
    ## [2,]  0.6379357
    ## [3,]  0.8768861
    ## [4,]  1.6320599
    ## [5,]  0.9864113
    ## 
    ## [[4]]
    ## [1] 0.8487958
