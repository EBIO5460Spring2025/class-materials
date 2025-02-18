#' ---
#' title: "Ant data: boosted regression tree"
#' author: Brett Melbourne
#' date: 20 Feb 2024
#' output:
#'     github_document
#' ---

#' Boosted regression tree illustrated with the ants data.

#+ results=FALSE, message=FALSE, warning=FALSE
library(ggplot2)
library(gridExtra) #arranging multiple plots
library(viridisLite)
library(dplyr)
library(tree)
library(gbm)


#' Ant data with 3 predictors of species richness

ants <- read.csv("data/ants.csv") |> 
    select(richness, latitude, habitat, elevation) |> 
    mutate(habitat=factor(habitat))

#' **Boosting** can be viewed as an ensemble prediction method that fits
#' successive, potentially shrunk, models to the residuals. The final prediction
#' is the sum of the models (we can alternatively view it as a weighted
#' average of the models).
#' 

#' A boosted regression tree algorithm:
#' ```
#' load y, x, xnew
#' set parameters: mincut, ntrees, lambda
#' set f_hat(xnew) = 0
#' set r = y (residuals equal to the data)
#' for m in 1 to ntrees
#'     train tree model on r and x
#'     predict residuals, r_hat_m(x), from trained tree  
#'     update residuals: r = r - lambda * r_hat_m(x)
#'     predict y increment, f_hat_m(xnew), from trained tree
#'     update prediction: f_hat(xnew) = f_hat(xnew) + lambda * f_hat_m(xnew)
#' return f_hat(xnew)
#' ```
#' 

#' Code this algorithm in R
#'

#+ cache=TRUE, results=FALSE

# Boosted regression tree algorithm

# load y, x, xnew
y <- ants$richness
x <- ants[,-1]
# xnew will be a grid of new predictor values on which to form predictions:
grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=10))
# or it could be set to the original x data:
# grid_data <- ants[,-1]

# Parameters
mincut <- 10 #Minimum size of decision nodes; controls tree complexity
ntrees <- 1000
lambda <- 0.01 #Shrinkage/learning rate/descent rate

# Set f_hat, r
f_hat <- rep(0, nrow(grid_data))
r <- y

ssq <- rep(NA, ntrees) #store ssq to visualize descent
for ( m in 1:ntrees ) {
#   train tree model on r and x
    data_m <- cbind(r, x)
    fit_m <- tree(r ~ ., data=data_m, mincut=mincut)
#   predict residuals from trained tree
    r_hat_m <- predict(fit_m, newdata=x)
#   update residuals (gradient descent)
    r <- r - lambda * r_hat_m
    ssq[m] <- sum(r ^ 2)
#   predict y increment from trained tree
    f_hat_m <- predict(fit_m, newdata=grid_data)
#   update prediction
    f_hat <- f_hat + lambda * f_hat_m
#   monitoring
    print(m)
}

# return f_hat
boost_preds <- f_hat


#' Plot predictions

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

#' Here's how the algorithm descended the loss function (SSQ)

plot(1:ntrees, ssq, xlab="Iteration (number of trees)")



#' Here is an animated version to visualize how the model changes over
#' iterations. Run this code to animate.
#+ eval=FALSE

# Animated boosted regression tree algorithm

# Pause for the specified number of seconds.
pause <- function( secs ) {
    start_time <- proc.time()
    while ( (proc.time() - start_time)["elapsed"] < secs ) {
        #Do nothing
    }
}

y <- ants$richness
x <- ants[,-1]
grid_data  <- expand.grid(
    latitude=seq(min(ants$latitude), max(ants$latitude), length.out=201),
    habitat=factor(c("forest","bog")),
    elevation=seq(min(ants$elevation), max(ants$elevation), length.out=10))

# Parameters
mincut <- 10 #Minimum size of decision nodes; controls tree complexity
ntrees <- 600
lambda <- 0.01 #Shrinkage/learning rate/descent rate

# Set f_hat, r
f_hat <- rep(0, nrow(grid_data))
r <- y

ssq <- rep(NA, ntrees) #store ssq to visualize descent
for ( m in 1:ntrees ) {
#   train tree model on r and x
    data_m <- cbind(r, x)
    fit_m <- tree(r ~ ., data=data_m, mincut=mincut)
#   predict residuals from trained tree
    r_hat_m <- predict(fit_m, newdata=x)
#   update residuals (gradient descent)
    r <- r - lambda * r_hat_m
    ssq[m] <- sum(r ^ 2)
#   predict y increment from trained tree
    f_hat_m <- predict(fit_m, newdata=grid_data)
#   update prediction
    f_hat <- f_hat + lambda * f_hat_m
    
#   animate
    resids <- data.frame(ants, residual=data_m$r)
    r_hat_grid <- predict(fit_m, newdata=grid_data)
    preds_r <- data.frame(grid_data, residual=r_hat_grid)
    preds_d <- cbind(grid_data, richness=f_hat)
    
#   animation can't be done with ggplot in real time; use base plotting.
    par(mfrow = c(2, 2), mar=c(1,0,0,1), oma=c(4,5,4,0))
    color_ramp <- colorRampPalette(viridis(100))
    elev_max <- 550
    #plot residuals
    for ( h in c("bog","forest") ) {
        resids_h <- subset(resids, habitat == h)
        colors <- color_ramp(10)[ceiling(10 * resids_h$elevation/elev_max)]
        plot(residual ~ latitude, col=colors, data=resids_h,
             ylim=c(-5, 20), pch=19, axes=FALSE)
        axis(1, labels=FALSE)
        if ( h == "bog" ) axis(2) else axis(2, labels=FALSE)
        box()
        mtext(h, line=1.5)
        if ( h == "bog" ) mtext("residuals", side=2, line=3)
        
      # Add a line for each elevation
      for (elev in unique(preds_r$elevation)) {
        preds_r_h <- subset(preds_r, elevation == elev & habitat == h)
        colors <- color_ramp(10)[ceiling(10 * preds_r_h$elevation/elev_max)]
        lines(residual ~ latitude, col=colors, data=preds_r_h, lty=2)
      }
    }
    #plot data
    for ( h in c("bog","forest") ) {
        ants_h <- subset(ants, habitat == h)
        colors <- color_ramp(10)[ceiling(10 * ants_h$elevation/elev_max)]
        plot(richness ~ latitude, col=colors, data=ants_h,
             ylim=c(0, 18), pch=19, axes=FALSE)
        axis(1)
        if ( h == "bog" ) axis(2) else axis(2, labels=FALSE)
        box()
        mtext("latitude", side=1, line=3)
        if ( h == "bog" ) mtext("richness", side=2, line=3)
        
        # Add a line for each elevation
        for (elev in unique(preds_d$elevation)) {
            preds_d_h <- subset(preds_d, elevation == elev & habitat == h)
            colors <- color_ramp(10)[ceiling(10 * preds_d_h$elevation/elev_max)]
            lines(richness ~ latitude, col=colors, data=preds_d_h, lty=2)
        }
    }
    mtext(paste("Iteration", m, "/", ntrees), outer=TRUE, line=2)
    pause(0.2)
}
boost_preds <- f_hat



#' Boosted regression trees are implemented in the gbm package

boost_ants1 <- gbm(richness ~ ., data=ants, distribution="gaussian", 
                  n.trees=1000, interaction.depth=1, shrinkage=0.01,
                  bag.fraction=1)
boost_preds <- predict(boost_ants1, newdata=grid_data)

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



