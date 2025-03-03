### Assignment 4

**Due:** Friday 14 Mar 11:59 PM

**Grading criteria:** Answer all the questions completely. On time submission.

**Percent of grade:** 7%




#### Random forest and boosting
**Learning goals:**

* Use mature random forest and boosting libraries with typical ecological data
* Use the state of the art xgboost library
* Explore parameters of the xgboost algorithm
* Understand the binomial (or cross entropy) loss function
* Understand the different types of errors in classification
* Practice implementing strategies for class imbalance
* Practice tuning machine learning algorithms using the CV inference algorithm
* Practice exploring different types of errors using a confusion matrix
* Practice comparing model performance using the CV inference algorithm with different performance measures
* Practice visualizing model predictions



Your ultimate goal is to train and tune random forest and gradient boosting models to predict the occurrence of a plant in New Zealand based on a range of predictor variables. The plant is "nz05" from an anonymized reference dataset (apparently NZ plants have better data-privacy protections than US citizens). These data are from one of the papers we'll read: 

* Valavi et al. (2021). Predictive performance of presence-only species distribution models: a benchmark study with reproducible code. *Ecological Monographs* 0:e01486. https://doi.org/10.1002/ecm.1486.

If you're curious, it could be super fun to expand on later in an individual project to see if you can improve the predictions in this paper.



#### New concepts

##### Classification case: Binomial (or cross entropy) loss function

In class we considered the regression case for boosting, where we used mean squared error as the loss function. For the classification case with two classes, it's natural to use the negative log-likelihood of the binomial distribution as the loss function. The likelihood for the binomial distribution is:

$$
L = \prod_{i=1}^n p_i^{y_i} (1 - p_i)^{1 - y_i}
$$

where $\prod$ is the product operator, $p$ is the predicted probability, $y$ is the value of the data (either 0 or 1), and $i$ indexes the $n$ data points. Taking logs, and multiplying by -1, we get the negative log likelihood:

$$
NLL = - \sum_{i=1}^n \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

Lower values of the negative log likelihood equate to lower loss. To train a classification model by maximum likelihood, we would minimize this negative log likelihood.

It turns out that the negative log likelihood is the same as the cross entropy. Entropy, *H*, is a concept from information theory. Entropy measures the amount of information in a random variable, *X*. It is defined as:

$$
H(X) = - \sum_x p(x) \log(p(x))
$$

where $p(x)$ is the probability that $X = x$. The cross entropy is closely related but instead compares two probability distributions, the true distribution $P$, and an estimated distribution $Q$, and is defined as:

$$
H(P, Q) = - \sum_x p(x) \log(q(x))
$$

For the case of binomial data (the true distribution) compared to the predicted probability (the estimated distribution), the cross entropy is:

$$
H(y, p) = - \sum_{i=1}^n \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

which is the same as the negative log likelihood. For the cross-entropy loss, we generally divide by the number of data points to get the average loss, much as we divide the total sum of squares by the number of data points to get the mean squared error. This allows us to compare datasets of different size on the same scale. So, the cross entropy loss is generally defined as the average entropy:

$$
\frac{1}{n} H(y, p)
$$

Thus, training a classification model by maximizing the binomial likelihood or minimizing the cross-entropy loss is equivalent. This is more than a coincidence. The maximum likelihood of the binomial distribution, maximizes the information we can gain from binary data. Here, we use this loss function to train our two-class boosting model. The above extends naturally to more than two classes to give the multinomial likelihood and categorical cross entropy. In different machine learning libraries, you might find this loss function variously called names that relate either to the binomial or multinomial likelihood or to cross entropy.



##### Confusion matrix and different types of errors and accuracies

Read about some of these concepts in James et al 4.4.2, starting from about paragraph 7.

So far, we have considered a simple metric of prediction performance: the simple prediction error rate:

$$
\mathrm{Error \ rate} = \frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat{y}_i)
$$

where $I$ is the indicator function that equals 1 when $y_i \neq \hat{y}_i$ and 0 otherwise. This is the overall error rate in a classification task. But it might not be the most relevant error. For example, we might be more concerned about making the error of falsely predicting an absence when a species is actually present than we are about making an error of falsely predicting a presence when a species is actually absent.  The various ways of making an error relate to the **confusion matrix**, a table of predicted to observed classes, and there are a myriad of terms in the literature that refer to the different types of errors or accuracies in different fields of study. Here's an example confusion matrix for presence-absence data containing the number of each case (with totals in the margins).

|                       | True presence | True absence | Total |
| --------------------- | ------------- | ------------ | ----- |
| **Predicted present** | 216           | 526          | 742   |
| **Predicted absent**  | 98            | 2872         | 2970  |
| **Total**             | 314           | 3398         | 3712  |

The overall error rate is the ratio of the number incorrectly classified out of the total:

$$
\mathrm{Error \ rate} = \frac{526 + 98}{3712} = 0.17
$$

But there are many other ratios in this table that we might be interested in considering, depending on our goal. These ratios are formed over different rows or columns of the confusion matrix.

The first two ratios are two types of prediction errors that we can make. The error of a false absence (or "false negative") is when we falsely predict a true presence to be absent. In this example, we see the false absence rate is much higher than the overall error rate:

Alternate names: **False absence rate**. **False negative rate**.

$$
\mathrm{False \ absence \ rate} = \frac{\mathrm{Number \ of \ falsely \ predicted \ absences}}{\mathrm{Number \ of \ true \ presences}} = \frac{98}{314} = 0.31
$$

The error of a false presence (or "false positive") is when we falsely predict a true absence to be present:

Alternate names: **False presence rate**. **False positive rate**.

$$
\mathrm{False \ presence \ rate} = \frac{\mathrm{Number \ of \ falsely \ predicted \ presences}}{\mathrm{Number \ of \ true \ absences}} = \frac{526}{3398} = 0.15
$$

The error rates above are formed as ratios over the columns. The complementary ratios, also formed over the columns, are the accuracy rates (1 - error rate): the true presence ("true positive") rate  (the proportion of true presences correctly predicted) , and the true absence ("true negative") rate (the proportion of true absences correctly predicted). In biomedicine and epidemiology, where accuracies of medical tests are crucial, these ratios are known as the sensitivity (ability of a test to detect a disease when present) and specificity (ability of a test not to be confused with other diseases). In machine learning, the true positive rate is also called the recall (ability of a model to "recall" or recognize the true class when it appears).

Alternate names: **True presence rate**. **True positive rate**. **Sensitivity**. **Recall**.

$$
\mathrm{True \ presence \ rate} = \frac{\mathrm{Number \ of \ true \ predicted \ presences}}{\mathrm{Number \ of \ true \ presences}} = \frac{216}{314} = 0.69
$$

Alternate names: **True absence rate**. **True negative rate**. **Specificity**.

$$
\mathrm{True \ absence \ rate} = \frac{\mathrm{Number \ of \ true \ predicted \ absences}}{\mathrm{Number \ of \ true \ absences}} = \frac{2872}{3398} = 0.85
$$

Similarly, ratios formed along the rows of the confusion matrix could be of interest. In particular, in machine learning the **precision** is the ability of a model to focus on true positives (best when all of the predicted presences are actually presences).

Alternate names: **Precision**. **Positive predictive value**.

$$
\mathrm{Precision} = \frac{\mathrm{Number \ of \ true \ predicted \ presences}}{\mathrm{Number \ of \ predicted \ presences}} = \frac{216}{742} = 0.29
$$

The precision is generally for the positive class in a positive vs negative classification task but the principle applies to other classes. The precision of the negative class (ability of a model to focus on true negatives) in the above table would be formed over the other row:

$$
\mathrm{Negative \ class \ precision} = \frac{\mathrm{Number \ of \ true \ predicted \ absences}}{\mathrm{Number \ of \ predicted \ absences}} = \frac{2872}{2970} = 0.97
$$



##### Class imbalance

The overall error rate above is also influenced by the number of instances of each class in the dataset. In many datasets, different classes are not represented evenly. This is known as class imbalance. Class imbalance occurs for most species' presence-absence data; there are typically many more absences than there are presences.  If there are a very large number of absences and only a small number of presences, it's easy to be accurate overall by simply always predicting absent. For example if the ratio of presences to absences is 1:100, then by always predicting absent we're assured of an error rate of only 1%, yet our error rate on the presence class (i.e. false absence rate) is 100%. If we tune our model to target the overall error rate, the model could have poor performance on other types of errors.

More generally, the smaller number of presences means the learning algorithm is not presented with as many presence cases to learn from and distinguish between classes, leading to poor predictive performance for that class. We'll consider several strategies here to alleviate class imbalance: 1) weighting observations by class representation, 2) sampling data more evenly between classes, 3) using a performance metric that targets the minority class or balances performance between the classes.



##### F1 and AUC performance metrics

The F1 performance metric emphasizes the correct classification of the minority positive class in a positive-negative classification task. This is the situation we are frequently in for presence-absence data. Recall and precision are the two attributes of the positive class. F1 is the harmonic mean of the recall and precision:

$$
F1 = 2 \frac{\mathrm{Recall} \times \mathrm{Precision}}{\mathrm{Recall} + \mathrm{Precision}}
$$

This approaches 1 for perfect recall and precision. From the example confusion matrix above,

$$
F1 = 2 \frac{0.69 \times 0.29}{0.69 + 0.29} = 0.41
$$

which is quite low. We could tune our models to increase this metric and balance the two prediction attributes of the presence class.

So far, we have also only considered using a threshold probability of 0.5 (the theoretically best or decision boundary, or Bayes classifier) to determine the predicted class. But using a **different threshold** instead might help to alleviate some of the concerns discussed above. What threshold to use instead will depend on the situation and requires domain knowledge, such as consideration of costs of different kinds of errors. One often-used strategy is to first find a model that has good performance across a range of thresholds, which leads to the **AUC performance metric** (the area under the ROC curve). The first step is to plot the **true positive rate** (sensitivity; recall) against the **false positive rate** over all possible thresholds. This yields what is known as the "receiver operating characteristics" curve or **ROC curve** (see Fig 4.8 of James et al). The name reflects it's origins in signal processing. The area under the curve (**AUC**) of the ROC curve summarizes the performance of a model over all thresholds.



#### Starter code

Here's some code for reading in the data, fitting models with random forest, gbm, and xgboost, and plotting predictions.

```R
library(disdat) #data package
library(dplyr)
library(ggplot2)
library(sf)
library(randomForest)
library(gbm)
library(xgboost)
library(precrec) #for AUC
```



Load presence-absence data for species "nz05"

```R
nz05df <- bind_cols(select(disPa("NZ"), nz05), disEnv("NZ")) |> 
    rename(occ=nz05)
head(nz05df)
```



Outline of New Zealand

```R
nzpoly <- disBorder("NZ")
class(nzpoly) #sf = simple features; common geospatial format
```



Plot presence (1) absence (0) data

```R
nz05df |> 
    arrange(occ) |> #place presences on top
    ggplot() +
    geom_sf(data=nzpoly, fill="lightgray") +
    geom_point(aes(x=x, y=y, col=factor(occ)), shape=1, alpha=0.2) +
    theme_void()
```



Data for modeling

```R
nz05pa <- nz05df |> 
    select(!c(group,siteid,x,y,toxicats))
head(nz05pa)
```



For demonstration, I'll use a simple train-test split. You'll do k-fold CV.

```R
set.seed(1234)
n <- nrow(nz05pa)
train_i <- sample(1:n, size=trunc(n * 0.8))
nz05pa_train <- nz05pa[train_i,]
nz05pa_test <- nz05pa[-train_i,]
```



Example basic random forest model (10 secs)

```R
# Prepare data for random forest
nz05pa_train_rf <- nz05pa_train
nz05pa_train_rf$occ <- factor(nz05pa_train_rf$occ) #for classification

# Train
nz05_train <- randomForest(occ ~ ., data=nz05pa_train_rf, ntree = 500)

# Predict on the test data
nz05_prob <- predict(nz05_train, newdata=nz05pa_test[,-1], type="prob")[,2]
```



Example random forest model incorporating a strategy for class imbalance. This model draws equal samples of presence and absence data at each bootstrap step to form each tree in the ensemble. The idea is that equal sampling instead of proportional sampling should produce trees that are not biased toward predicting the absence class.

```R
# Train
n_pres <- sum(nz05pa_train$occ)
nz05_train <- randomForest(occ ~ ., data=nz05pa_train_rf, ntree = 500,
                           sampsize=c(n_pres, n_pres))

# Predict on the test data
nz05_prob <- predict(nz05_train, newdata=nz05pa_test[,-1], type="prob")[,2]
```



Example basic boosted model with gbm (1.5 mins). It's a good idea to use gbm as a sanity check for xgboost.

```R
# Train
nz05_train <- gbm(occ ~ ., data=nz05pa_train, distribution="bernoulli", 
                  n.trees=10000, interaction.depth=3, shrinkage=0.01, 
                  bag.fraction=0.5)

# Predict on the test data
nz05_prob <- predict(nz05_train, newdata=nz05pa_test[,-1], type="response")

```



Example basic boosted model with xgboost (30-60 secs). This model has approximately the same specification as gbm above. It seems to perform about as well or a little better, confirming that we probably have xgboost working as intended.

```R
# Prepare data for xgboost
nz05pa_train_xgb <- xgb.DMatrix(data=as.matrix(nz05pa_train[,-1]), 
                                label=nz05pa_train$occ)
class(nz05pa_train_xgb)
colnames(nz05pa_train_xgb)

# Train
nz05_train <- xgboost(data=nz05pa_train_xgb, eta=0.01, max.depth=3, 
                      subsample=0.5, nrounds=10000, print_every_n=1000, 
                      nthread=2, objective="binary:logistic")

# Predict on the test data
nz05pa_test_xgb <- xgb.DMatrix(data=as.matrix(nz05pa_test[,-1]))
nz05_prob <- predict(nz05_train, newdata=nz05pa_test_xgb)
```



Example xgboost model incorporating a strategy for class imbalance. This model weights the presence data, essentially so that the learning rate is higher for the presences. Here I set the weighting parameter, `scale_pos_weight`, equal to the ratio of absences to presences. This is often a good starting point but this is a tunable parameter.

```R
# Train
n_pres <- sum(nz05pa_train$occ)
n_abs <- sum(nz05pa_train$occ == 0)
ab_pr_ratio <-  n_abs / n_pres
nz05_train <- xgboost(data=nz05pa_train_xgb, eta=0.01, max.depth=3, 
                      subsample=0.5, nrounds=10000, print_every_n=1000, 
                      scale_pos_weight=ab_pr_ratio, nthread=2,
                      objective="binary:logistic")

# Predict on the test data
nz05pa_test_xgb <- xgb.DMatrix(data=as.matrix(nz05pa_test[,-1]))
nz05_prob <- predict(nz05_train, newdata=nz05pa_test_xgb)
```



The models above predict probability. Convert to predict presence(1) and absence(0):

```R
threshold <- 0.5
nz05_pred <- 1 * (nz05_prob > threshold)
```



Characteristics of a prediction

```R
hist(nz05_prob)
max(nz05_prob)
sum(nz05_prob > threshold) #number of predicted presences

table(nz05_pred, nz05pa_test$occ)  #confusion matrix
mean(nz05_pred == nz05pa_test$occ) #accuracy
mean(nz05_pred != nz05pa_test$occ) #error = 1 - accuracy

F1_score <- function(predicted_class, true_class) {
    confusion <- table(predicted_class, true_class)
    recall <- confusion[2,2] / sum(confusion[,2])
    precision <- confusion[2,2] / sum(confusion[2,])
    score <- 2 * recall * precision / (recall + precision)
    return(score)
}

F1_score(nz05_pred, nz05pa_test$occ)

ROC_curve <- evalmod(scores=nz05_pred, labels=nz05pa_test$occ)
autoplot(ROC_curve, curvetype="ROC", type="p")
AUC_score <- auc(ROC_curve)[1,4]
AUC_score
```



Example prediction for a grid of the predictor variables mapped across NZ with 2 km pixels. I prepared a grid of the predictor variables and saved it to a `.csv` file (find it in the `data` folder; preparation details are in the associated "about" file). The file contains the values of the predictor variables for each spatial location (x, y). The grid has a lower spatial resolution than the original rasters (which are massive) but still plenty of resolution (2 km pixels) to make a nice map.

```R
# Read in the grid of predictor variables
NZ_grid <- read.csv("data/NZ_predictors.csv")
head(NZ_grid)

# Prepare data for xgboost (exclude spatial location columns)
NZ_grid_xgb <- xgb.DMatrix(data=as.matrix(select(NZ_grid, !c(x,y))))
colnames(NZ_grid_xgb)

#Predict
nz05_grid_prob <- predict(nz05_train, newdata=NZ_grid_xgb)
nz05_grid_present <- 1 * (nz05_grid_prob > 0.5)

# Map probability prediction
NZ_grid |>
    bind_cols(prob=nz05_grid_prob) |>
    ggplot() +
    geom_tile(aes(x=x, y=y, fill=prob)) +
    scale_fill_viridis_c() +
    coord_equal() +
    theme_void() +
    labs(fill = "Probability")
    
# Map presence prediction
NZ_grid |>
    bind_cols(present=nz05_grid_present) |>
    ggplot() +
    geom_tile(aes(x=x, y=y, fill=factor(present))) +
    coord_equal() +
    theme_void() +
    labs(fill = "Present")
```



**Q1\.**  Use 5-fold cross validation to estimate overall accuracy, F1 score, and AUC for the basic random forest model above. Don't bother tuning the model. 

**Q2\.**  Use 5-fold cross validation to estimate overall accuracy, F1 score, and AUC for the class imbalance random forest model above. Again, don't bother tuning the model. How much better or worse does this model perform on the metrics? This is your benchmark model. Can you beat this model with xgboost?

**Q3\.** Train and tune a gradient boosting model across the six boosting parameters ( `eta`, `max_depth`, `subsample`, `colsample_bytree`, `nrounds`, `scale_pos_weight`). The goal is to find a model with excellent predictive performance (not necessarily "the best") and especially to beat the random forest benchmark above.

Suggestions:

* Think about each of the xgboost parameters and if they have counterparts in the random forest model. Start by setting them to be roughly equivalent to the random forest model.
* Use gbm if needed as a sanity check for xgboost
* Use a grid search
* Use 5-fold CV
* Use parallel processing, which in `xgboost` is best done using `xgboost` itself. This is super easy. Just set the `nthread` argument to a number greater than 1. I suggest using a supercomputer node. Higher-level parallel processing on the grid search is not helpful (according to xgboost documentation).
* Tune models to maximize the F1 score

**Q4\.** Like other tree ensemble methods we have looked at, `xgboost` can provide information about the relative importance of the different predictor variables. Research how to do this and make a plot that displays this information.

**Q5\.** Plot the prediction both as a presence map and a map of probabilities.
