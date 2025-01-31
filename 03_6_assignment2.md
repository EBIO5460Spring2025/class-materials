### Assignment 2

**Due:** Thursday 13 Feb 11:59 PM

**Grading criteria:** Answer all the questions completely (including all the check boxes). On time submission.

**Percent of grade:** 7%

**Format for submitting assignments:**

* Submit code and answers to questions as comments in the same script or markdown report.
* All code and text in one file please (i.e. not a separate file per question).
* Please include all the produced plots along with your code. I don't want to run your code to make plots!! You may produce a report if you wish (e.g. markdown, html, pdf) but a script file plus plots is sufficient.
* Please organize your repository with files clearly labelled to indicate belonging to the assignment, or put each assignment in its own directory (folder).



#### Code a classification prediction task

This example is a species distribution model

**Learning goals:**

* Understand the steps of the KNN model algorithm
* Practice modifying a function to examine its internal workings
* Practice visualizing model outputs
* Understand how classification model algorithms capture nonlinear relationships with predictor variables through a probability function
* Understand the bias-variance tradeoff in the context of classification model algorithms
* Practice tuning a machine learning algorithm using the CV inference algorithm
* Practice the basic tasks needed to produce a species distribution map



The goal of this assignment is to find the best performing KNN model to predict the presence or absence of the Willow Tit, a bird native to Switzerland, using elevation as a predictor. This data set is from Royle JA & Dorazio RM (2008) Hierarchical Modeling and Inference in Ecology, p 87. Elevation (altitude) data are from [USGS-EROS](https://www.usgs.gov/centers/eros). To focus on understanding how the KNN model algorithm expresses a nonlinear function of the predictors, we'll consider a single predictor variable.

The code below is R. Corresponding Python code is in `assignment2.py`.

The data are in the `data` directory:

```R
library(ggplot2)
library(dplyr)

# Read in the data
willowtit <- read.csv("data/wtmatrix.csv") |> 
    mutate(status=ifelse(y.1==1, "present", "absent")) |> 
    select(status, elev)
head(willowtit)

# Summary
willowtit |> 
    group_by(bin=cut(elev, breaks=seq(0, 3000, by=500), dig.lab=4), status) |> 
    summarize(p=n()) 
```



**Q1\.** Based on the code in `classification_knn.R` that we worked through in class:

- [ ] Form predictions (present or absent) for the KNN model for various values of `k_knn` (number of nearest neighbors).

- [ ] Visualize these predictions as a function of elevation in a scatterplot. How might you visualize these? Would you choose to plot them as points? or bars? or ... This requires some thought as the raw predictions are category labels (a character vector).

**Q2.** Make a new function, called `knn_classify2_probs`, by modifying the `knn_classify2` function so that instead of returning a character vector of the predictions, it returns a numeric vector of the probabilities. To do this, you should work through the existing function line-by-line to understand its flow. Hint: the solution is to remove some lines of code. Also, the KNN function predicts the probability of `cat1`, which in this case could be absence, so be sure to include code that returns always (in the general case, not just this dataset) the probability of presence.

- [ ] Include updated documentation for this function, i.e. don't just leave the original documentation (comments) because that doesn't apply any more!
- [ ] Use this function to form predicted probabilities for various values of `k_knn`. Visualize these predictions in a scatterplot of probability as a function of elevation. Specifically, make plots for `k_knn` = 1, 5, 25, 100.
- [ ] Describe how the shape of the function changes with `k_knn`.
- [ ] Comment on how the value of `k_knn` influences the flexibility of the function and it's likely bias and variance (i.e. consider the bias-variance tradeoff in the context of classification); e.g. one of these models is likely a better balance between bias and variance because it represents the prediction smoothly yet relatively closely.

**Q3.** Tune the KNN model algorithm using the CV inference algorithm. Specifically, using the original `knn_classify2` function, and based on the code in `classification_knn.R`, find the best predictive KNN model using 5, 10, and n-fold CV. Hint: be sure to search adequately across the possible range of `k_knn`. You want to see the error go down, and then go back up again. If the error is not obviously changing much, you've probably got more parameter space to explore.

- [ ] Comment on the stability of the CV plot for a single k-fold run, i.e. how does it vary across runs?
- [ ] Investigate a multiple replicate run of CV for 5, 10, and n-fold CV to stabilize the plot (this may take a while to run)
- [ ] What value of `k_knn` would you choose for good predictive performance? Justify your choice.

**Q4.** Visualize probability versus elevation for the best model.

- [ ] Comment on the shape of this relationship
- [ ] Color code the two categories (presence, absence) in this plot and add a horizontal line at probability = 0.5. From the plot, what is the approximate elevation range of predicted presence?
- [ ] Why are there no predicted presences below probability = 0.5?

**Q5.** Make a map of where the Willow Tit is predicted to be.

- [ ] Do the predictions make sense? Briefly describe why the spatial pattern of predictions in the map makes sense.



To get you started on making a map, here is code to read in and plot elevation data (digital elevation model, DEM) for Switzerland.

```R
swissdem <- read.csv("data/switzerland_tidy.csv")

ggplot(swissdem) +
    geom_raster(aes(x=x, y=y, fill=Elev_m)) +
    scale_fill_gradientn(colors=terrain.colors(22), name="Elevation (m)") + 
    coord_quickmap() +
    labs(title="Switzerland: DEM") +
    theme_void() +
    theme(plot.title=element_text(hjust=0.5, vjust=-2))
```

Here is how we could overlay the predicted presences of the Willow Tit on the DEM. First add a column to `swissdem` indicating the **predicted** presence/absence status. Call the column, say, `pred_status`. To provide working code *for illustration only*, I've hard coded an elevation range here just so we have a vector called `pred_status` to plot onto a map.

```R
swissdem <- swissdem |> 
    mutate(pred_status=ifelse(Elev_m > 800 & Elev_m < 1200, "present", "absent"))
```

You wouldn't use this code at all! Instead, add a `pred_status` column of model predictions to `swissdem` containing the predicted presence-absence status (e.g. using `cbind()`).  To do that, you'll need to first feed the elevations in `swissdem` to the KNN model to make the predictions.

Now add the predicted status using a `geom_tile()` layer to make an overlay. We can filter to just the presences so absences are blank, and use an `alpha` setting  of say, 0.6, for some transparency. The overlaid presences will appear in the map as a light shade of blue.

```R
ggplot() +
    geom_raster(data=swissdem,
                aes(x=x, y=y, fill=Elev_m)) +
    scale_fill_gradientn(colors=terrain.colors(22), name="Elevation (m)") +
    geom_tile(data=filter(swissdem, pred_status=="present"), 
              aes(x=x, y=y), fill="blue", 
              alpha=0.6) +
    coord_quickmap() +
    labs(title="Predicted distribution of Willow Tit in Switzerland") +
    theme_void() +
    theme(plot.title=element_text(hjust=0.5, vjust=-2))
```

 You might try other color schemes. For example, `scale_fill_viridis_c()` with a pink overlay looks pretty good to me too.



**Optional advanced ggplot for R:** The above default `ggplot` does not include a legend or color scale for presence-absence. There is a bit of extra work to add legends for both elevation and presence-absence. Here is one way of doing it using `scale_fill_manual()` and the package `ggnewscale`.

```r
library(ggnewscale)

ggplot() +
    geom_raster(data=swissdem,
                aes(x=x, y=y, fill=Elev_m)) +
    scale_fill_gradientn(colors=terrain.colors(22), name="Elevation (m)") +
    new_scale_fill() +
    geom_tile(data=filter(swissdem, status=="present"), 
              aes(x=x, y=y, fill="Present"), 
              alpha=0.6) +
    scale_fill_manual(name="Willow Tit", 
                      breaks=c("Present"), 
                      values=c("Present"="blue")) +
    coord_quickmap() +
    labs(title="Predicted distribution of Willow Tit in Switzerland") +
    theme_void() +
    theme(plot.title=element_text(hjust=0.5, vjust=-2))
```



**Push your code to GitHub**
