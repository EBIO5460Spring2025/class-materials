# Classification in machine learning
# Brett Melbourne
# 1 Feb 2024

# This example is from Chapter 2.2.3 of James et al. (2021). An Introduction to
# Statistical Learning. It is the simulated dataset in Fig 2.13.

import pandas as pd
import numpy as np
from plotnine import *
from itertools import product

# Create a data frame with a grid of all combinations of the variables specified
# in grid_dict.
#     https://pandas.pydata.org/pandas-docs/version/0.17.1/
#     cookbook.html#creating-example-data
# grid_dict: the variables to make the grid (dictionary)
# return: combinations (dataframe)
#
def expand_grid(grid_dict):
    rows = product(*grid_dict.values())
    return pd.DataFrame.from_records(rows, columns=grid_dict.keys())

# Orange-blue data:

orbludat = pd.read_csv("data/orangeblue.csv")

(ggplot(orbludat)
 + geom_point(aes(x="x1", y="x2", color="category"),
              fill="#00000000", size=2, stroke=0.3)
 + scale_color_manual(values=["blue", "orange"])
 + theme(panel_grid=element_blank()))


# KNN function for a data frame of x_new
# x:       x data of variables in columns (dataframe, numeric)
# y:       y data, 2 categories (dataframe, character)
# x_new:   values of x variables at which to predict y (dataframe, numeric)
# k:       number of nearest neighbors to average (scalar, integer)
# rng:     numpy random generator, set ahead rng = np.random.default_rng()
#
# return:  predicted y at x_new (dataframe, character)
#
def knn_classify2(x, y, x_new, k, rng):
    x_new = np.array(x_new)
    n = len(x_new)
    df = pd.DataFrame()
    category = y.unique() #get the two category names
    df["y_int"] = np.where(y == category[0], 1, 0) #convert category to integer
    # Estimate probability of category 1 for each row of x_new
    p_cat1 = np.full(n, np.nan)
    for i in range(n):
    #   Distance of x_new to other x (Euclidean, i.e. sqrt(a^2+b^2+..))
        df["d"] = np.sqrt(np.sum((x - x_new[i])**2, axis=1))
    #   Sort y ascending by d; break ties randomly
        df["ran"] = rng.random(len(df))
        sorted_df = df.sort_values(by=["d","ran"])
    #   Mean of k nearest y data (frequency of category 1)
        p_cat1[i] = sorted_df["y_int"][:k].mean()
    # Predict the categories
    y_pred = np.where(p_cat1 > 0.5, category[0], category[1])
    # Break ties if probability is equal (i.e. exactly 0.5)
    rnd_category = rng.choice(category, n, replace=True) #vector of random labels
    tol = 1 / (k * 10)  # tolerance for checking equality
    y_pred = np.where(np.abs(p_cat1 - 0.5) < tol, rnd_category, y_pred)
    return pd.DataFrame(y_pred, columns=["category"])


# Test the output of the knn_classify2 function
rng = np.random.default_rng() #start random number generator
nm = pd.DataFrame(rng.random((4, 2)), columns=["x1","x2"])
knn_classify2(orbludat[['x1', 'x2']], orbludat['category'], nm, k=10, rng=rng)

# Plot
grid_x = expand_grid( {"x1": np.arange(0, 1.01, 0.01),
                       "x2": np.arange(0, 1.01, 0.01)})
pred_category = knn_classify2(x=orbludat[['x1', 'x2']], y=orbludat['category'],
                              x_new=grid_x, k=2, rng=rng)
preds = pd.concat([grid_x, pred_category], axis=1)

(ggplot(orbludat)
 + geom_point(aes(x="x1", y="x2", color="category"),
              fill="#00000000", size=2, stroke=0.4)
 + geom_point(aes(x="x1", y="x2", color="category"), data=preds,
              shape=".", size=0.1)
 + scale_color_manual(values=["blue", "orange"])
 + theme_bw()
 + theme(panel_grid=element_blank()))


# k-fold CV for KNN. Be careful not to confuse the k's!

# Function to divide a data set into random partitions for cross-validation
# n:       length of dataset (scalar, integer)
# k:       number of partitions (scalar, integer)
# rng:     numpy random generator, set ahead rng = np.random.default_rng()
# return:  partition labels (vector, integer)
# 
def random_partitions(n, k, rng):
    min_n = n // k
    extras = n - k * min_n
    labels = np.concatenate([np.repeat(np.arange(1, k+1), min_n), 
             np.arange(1, extras+1)])
    partitions = rng.choice(labels, n, replace=False)
    return partitions



# Function to perform k-fold CV for the KNN model algorithm on the orange-blue
# data from James et al. Ch 2.
# k_cv:    number of folds (scalar, integer)
# k_knn:   number of nearest neighbors to average (scalar, integer)
# return:  CV error as error rate (scalar, numeric)
#
def cv_knn_orblu(k_cv, k_knn):
    orbludat_partition = random_partitions(len(orbludat), k_cv, rng)
    e = np.full(k_cv, np.nan)
    for i in range(0, k_cv):
        test_data = orbludat[orbludat_partition == i + 1]
        train_data = orbludat[orbludat_partition != i + 1]
        pred = knn_classify2(x=train_data[['x1', 'x2']],
                             y=train_data['category'],
                             x_new=test_data[['x1', 'x2']],
                             k=k_knn,
                             rng=rng)
        errors = np.array(pred["category"]) != np.array(test_data["category"])
        e[i] = np.mean(errors)
    cv_error = np.mean(e)
    return cv_error



# Use/test the function

cv_knn_orblu(k_cv=10, k_knn=10)
cv_knn_orblu(k_cv=len(orbludat), k_knn=10) #LOOCV


# Explore a grid of values for k_cv and k_knn

grid = expand_grid( {"k_cv": [5,10,len(orbludat)],
                     "k_knn": range(1,17)})
cv_error = np.full(len(grid), np.nan)
rng = np.random.default_rng(seed=6456) #For reproducible results
for i in range(len(grid)):
    cv_error[i] = cv_knn_orblu(grid["k_cv"][i], grid["k_knn"][i])
    print(round(100 * i  / len(grid))) #monitoring
cv_error = pd.DataFrame(cv_error, columns=["cv_error"])
result1 = pd.concat([grid, cv_error], axis=1)

# Plot the result.

(ggplot(result1)
+ geom_line(aes(x="k_knn", y="cv_error", color="factor(k_cv)"))
+ labs(color="k_cv"))


# LOOCV and 5-fold CV with many replicate CV runs with random partitions. This
# will take about an hour.

grid = expand_grid( {"k_cv": [5, len(orbludat)],
                     "k_knn": range(1, 16+1)})
reps = 250
cv_error = np.full((len(grid), reps), np.nan)
rng = np.random.default_rng(seed=8031) #for reproducible results
for j in range(reps):
    for i in range(len(grid)):
        cv_error[i,j] = cv_knn_orblu(grid["k_cv"][i], grid["k_knn"][i])
    print(round(100 * j / reps)) #monitor
mean_cv = np.mean(cv_error, axis=1)    
mean_cv = pd.DataFrame(mean_cv, columns=["cv_error"])
result2 = pd.concat([grid, mean_cv], axis=1)

# Plot the result
titlelab = f"Mean across {reps} k-fold CV runs"
(ggplot(result2)
+ geom_line(aes(x="k_knn", y="cv_error", color="factor(k_cv)"))
+ labs(title=f"Mean across {reps} k-fold CV runs", color="k_cv"))

# Print out the detailed numbers:

result2
