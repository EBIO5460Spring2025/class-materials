#' ---
#' title: "Ant data: xgboost"
#' author: Brett Melbourne
#' date: 20 Feb 2024
#' output:
#'     github_document
#' ---

#' xgboost illustrated with the ants data.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(dplyr)
#library(tree)
library(xgboost)

#' xgboost is a very fast implementation of boosted regression trees with
#' several innovations on top of basic gradient boosting. It natively supports
#' parallel CPU and GPU training. It is widely regarded as the current state of
#' the art.
#' 

#' Ant data with 3 predictors of species richness. xgboost takes only numeric
#' data, so we need to encode habitat with a dummy variable, which will be the
#' new column "forest".

ants <- read.csv("data/ants.csv") |> 
    mutate(forest=ifelse(habitat == "forest", 1, 0)) |>
    select(richness, latitude, elevation, forest, habitat)
ants

#' We need an xgboost matrix as the data structure to pass to xgboost. This
#' specifies the predictor variables and the response variable.

dtrain <- xgb.DMatrix(data=as.matrix(ants[,2:4]), label=ants$richness)
class(dtrain)
colnames(dtrain)

#' Train the model (eta is the learning rate)
#+ results=FALSE

bstDMatrix <- xgboost(data=dtrain, max_depth=2, eta=0.01, nthread=2, 
                      nrounds=1000, objective="reg:squarederror")

#' Predictions from the model

grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=51),
    forest=0:1)
boost_preds <- predict(bstDMatrix, newdata=as.matrix(grid_data))

#' Plot. This is a very expressive model.
                      
grid_data$habitat <- ifelse(grid_data$forest == 1, "forest", "bog")
preds <- cbind(grid_data, richness=boost_preds)
ants |>
    ggplot() +
    geom_line(data=preds, 
              aes(x=latitude, y=richness, col=elevation, group=factor(elevation)),
              linetype=2) +
    geom_point(aes(x=latitude, y=richness, col=elevation)) +
    facet_wrap(vars(habitat)) +
    scale_color_viridis_c() +
    theme_bw()
