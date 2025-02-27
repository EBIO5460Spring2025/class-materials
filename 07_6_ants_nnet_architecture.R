#' ---
#' title: "Ant data: neural network architectures"
#' author: Brett Melbourne
#' date: 28 Feb 2024
#' output:
#'     github_document
#' ---

#' Different neural network architectures illustrated with the ants data using
#' Keras (tensorflow). We compare a wide to a deep architecture.
#'
#' It's important to note that it isn't sensible to fit these 151 parameter
#' models to our small dataset of 44 data points without a lot of regularization
#' and of course tuning and k-fold cross validation, the latter of which would
#' add so much computation that it's not worth it. This code is to illustrate
#' the effect of different architectures and for comparison to the previous
#' machine learning approaches we have used with this small dataset.

#+ results=FALSE, message=FALSE, warning=FALSE
reticulate::use_condaenv(condaenv = "r-tensorflow")
library(ggplot2)
library(dplyr)
library(keras)
library(rgl) #3D plotting

#' Ant data with 3 predictors of species richness

ants <- read.csv("data/ants.csv") |> 
    select(richness, latitude, habitat, elevation)
head(ants)

#' Scaling parameters

lat_mn <- mean(ants$latitude)
lat_sd <- sd(ants$latitude)
ele_mn <- mean(ants$elevation)
ele_sd <- sd(ants$elevation)

#' Prepare the data and a set of new x to predict
 
xtrain <- ants |> 
    mutate(latitude = (latitude - lat_mn) / lat_sd,
           elevation = (elevation - ele_mn) / ele_sd,
           bog = ifelse(habitat == "bog", 1, 0),
           forest = ifelse(habitat == "forest", 1, 0)) |>    
    select(latitude, bog, forest, elevation) |>     #drop richness & habitat
    as.matrix()

ytrain <- ants[,"richness"]

grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=c("forest","bog"),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=51))

x <- grid_data |>
    mutate(latitude = (latitude - lat_mn) / lat_sd,
           elevation = (elevation - ele_mn) / ele_sd,
           bog = ifelse(habitat == "bog", 1, 0),
           forest = ifelse(habitat == "forest", 1, 0)) |>    
    select(latitude, bog, forest, elevation) |>     #drop richness & habitat
    as.matrix()


#' A wide model with 25 units

tensorflow::set_random_seed(6590)
modnn2 <- keras_model_sequential(input_shape = ncol(xtrain)) |>
    layer_dense(units = 25) |>
    layer_activation("relu") |> 
    layer_dense(units = 1)
modnn2

#+ eval=FALSE
compile(modnn2, optimizer="rmsprop", loss="mse")
fit(modnn2, xtrain, ytrain, epochs = 500, batch_size=4) -> history

#+ eval=TRUE
# save_model_tf(modnn2, "07_6_ants_nnet_architecture_files/saved/modnn2")
# save(history, file="07_6_ants_nnet_architecture_files/saved/modnn2_history.Rdata")
modnn2 <- load_model_tf("07_6_ants_nnet_architecture_files/saved/modnn2")
load("07_6_ants_nnet_architecture_files/saved/modnn2_history.Rdata")

#+ eval=TRUE
plot(history, smooth=FALSE, theme_bw=TRUE)

#+ eval=TRUE
npred <- predict(modnn2, x)
preds <- cbind(grid_data, richness=npred)
ants |> 
    ggplot() +
    geom_line(data=preds, 
              aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(aes(x=latitude, y=richness, col=elevation)) +
    facet_wrap(vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()

#' For this wide model, we get quite a flexible fit with a good deal of
#' nonlinearity and some complexity to the surface (e.g. the fold evident in the
#' bog surface).
#' 

#' Plot in 3D
#+ eval=FALSE

library(rgl)
cols <- ifelse(ants$habitat == "forest", "green", "brown")
plot3d(preds$latitude, preds$elevation, preds$richness)
points3d(ants$latitude, ants$elevation, ants$richness, col=cols)


#' A deep model with 25 units

tensorflow::set_random_seed(7855)
modnn3 <- keras_model_sequential(input_shape = ncol(xtrain)) |>
    layer_dense(units = 5) |>
    layer_activation("relu") |>
    layer_dense(units = 5) |>
    layer_activation("relu") |> 
    layer_dense(units = 5) |>
    layer_activation("relu") |> 
    layer_dense(units = 5) |>
    layer_activation("relu") |> 
    layer_dense(units = 5) |>
    layer_activation("relu") |> 
    layer_dense(units = 1)
modnn3

#+ eval=FALSE
compile(modnn3, optimizer="rmsprop", loss="mse")
fit(modnn3, xtrain, ytrain, epochs = 500, batch_size=4) -> history


#+ eval=TRUE
# save_model_tf(modnn3, "07_6_ants_nnet_architecture_files/saved/modnn3")
# save(history, file="07_6_ants_nnet_architecture_files/saved/modnn3_history.Rdata")
modnn3 <- load_model_tf("07_6_ants_nnet_architecture_files/saved/modnn3")
load("07_6_ants_nnet_architecture_files/saved/modnn3_history.Rdata")


#+ eval=TRUE
plot(history, smooth=FALSE, theme_bw=TRUE)

#+ eval=TRUE
npred <- predict(modnn3, x)
preds <- cbind(grid_data, richness=npred)
ants |> 
    ggplot() +
    geom_line(data=preds, 
              aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(aes(x=latitude, y=richness, col=elevation)) +
    facet_wrap(vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()

#' The deep model is very "expressive". It has more complexity to its fit, for
#' example more folds and bends in the surface, for the same number of
#' parameters and epochs. You can also see that this model is probably nonsense
#' overall given the many contortions it is undergoing to fit the data. It is
#' likely very overfit and unlikely to generalize well.
#' 

#' Plot in 3D
#+ eval=FALSE

cols <- ifelse(ants$habitat == "forest", "green", "brown")
plot3d(preds$latitude, preds$elevation, preds$richness)
points3d(ants$latitude, ants$elevation, ants$richness, col=cols)
