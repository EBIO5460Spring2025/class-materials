# Ant data: k-fold cross validation
# Brett Melbourne
# 23 Jan 2025

# Explore the cross-validation **inference algorithm** from scratch with the
# ants data and a polynomial model. Our goal is to predict richness of forest
# ants from latitude. What order of a polynomial **model algorithm** gives the
# most accurate predictions?

import pandas as pd
import numpy as np
from plotnine import *
pd.options.mode.copy_on_write = True #while in transition to pandas 3.0

# Ant data:

ants = pd.read_csv("data/ants.csv")

# Forest ant data:

forest_ants = ants[ants["habitat"] == "forest"]

ggplot(forest_ants) +\
    geom_point(aes(x="latitude", y="richness")) +\
    ylim(0,20)

# Here's one way we could code a 3rd order polynomial by first creating new
# variables for the quadratic (squared) and cubic (cubed) terms. 

forest_ants["latitude_2"] = forest_ants["latitude"] ** 2
forest_ants["latitude_3"] = forest_ants["latitude"] ** 3
forest_ants.head()

# We can then define the model algorithm using model formula notation with the
# `ols()` function from the `statsmodels` package, and train the model by
# minimizing the SSQ using the `fit()` method. This approach is substantially
# similar to R's `lm()`.

import statsmodels.formula.api as smf

poly_3 = smf.ols(formula="richness ~ latitude + latitude_2 + latitude_3", data=forest_ants)
poly_trained = poly_3.fit()
poly_trained.params

# We could alternatively define the model algorithm in terms of the X and y
# structures using the `LinearRegression()` function from the `scikit-learn`
# package, and train the model by minimizing the SSQ using the `fit()` method.

from sklearn.linear_model import LinearRegression

X = forest_ants[['latitude', 'latitude_2', 'latitude_3']]
y = forest_ants['richness']
model = LinearRegression()
model.fit(X, y)
model.coef_
model.intercept_

# This more general and flexible approach is needed to allow for more complex
# models, such as an orthogonal polynomial model. In `scikit-learn` there is a
# general approach to transforming variables. First, we define a transformer
# object (below: `ortho_poly`). We then estimate or set any parameters needed for
# the transformation with a `fit()` method and carry out the transformation with
# a `transform()` method; or these operations can be combined with
# `fit_transform()`. The parameters needed for the transformation are kept in
# the transformer object `ortho_poly`. This is important because we'll need to
# use the same transformation with the same parameters to form out-of-sample
# predictions with new X data. We then define and train the model as above and
# predict for new X values using the `predict()` method. There is 12 lines of
# code compared to the 6 lines of code in R.

# The following code trains the order 4 polynomial and plots the trained
# model. Use this block of code to try different values for the order of the
# polynomial. We can get up to order 21, which passes through every data point.

from source.transforms import Poly

order = 4 #integer
ortho_poly = Poly(degree=order)
X = ortho_poly.fit_transform(forest_ants['latitude'])
y = forest_ants['richness']
poly_model = LinearRegression()
poly_model.fit(X, y)
grid_latitude = np.linspace(min(forest_ants['latitude']), max(forest_ants['latitude']), 201)
new_df = pd.DataFrame(grid_latitude, columns=['latitude'])
new_X = ortho_poly.transform(new_df['latitude'])
pred_richness = poly_model.predict(new_X)
pred_richness = pd.DataFrame(pred_richness, columns=['richness'])
preds = pd.concat([new_df, pred_richness], axis=1)
poly_model.intercept_
poly_model.coef_

ggplot(aes(x='latitude', y='richness')) +\
    geom_point(data=forest_ants) +\
    geom_line(data=preds) +\
    coord_cartesian(ylim=(0,20)) +\
    labs(title=f"Polynomial order {order}")

# Use `predict` to ask for predictions from the trained polynomial model. For
# example, here we are asking for the prediction at latitude 43.2 and we find
# the predicted richness for the order=4 polynomial is 5.45. We need to provide
# the predictor variable `latitude` as a data frame even if it's just one value,
# and it needs to be first transformed.

new_df = pd.DataFrame([[43.2]], columns=['latitude'])
new_X = ortho_poly.transform(new_df['latitude'])
poly_model.predict(new_X)


########## Exploring the k-fold CV algorithm

# First, we need a function to divide the dataset up into partitions.

# Function to divide a data set into random partitions for cross-validation
# n:       length of dataset (scalar, integer)
# k:       number of partitions (scalar, integer)
# rng:     numpy random generator, set ahead rng = np.random.default_rng()
# return:  partition labels ranging from 0 to k-1 (vector, integer)
# 
def random_partitions(n, k, rng):
    min_n = n // k
    extras = n - k * min_n
    labels = np.concatenate([np.repeat(np.arange(k), min_n), 
             np.arange(extras)])
    partitions = rng.choice(labels, n, replace=False)
    return partitions

# To use this function, we first need to start the random number generator
rng = np.random.default_rng()

# What does the output of `random_partitions()` look like? It's a set of labels
# that says which partition each data point belongs to.
random_partitions(len(forest_ants), k=5, rng=rng)
random_partitions(len(forest_ants), k=len(forest_ants), rng=rng)

# Now code up the k-fold CV algorithm (from our pseudocode to Python code) to
# estimate the prediction mean squared error for one order of the polynomial.
# Try 5-fold, 10-fold, and n-fold CV. Try different values for polynomial
# order.

order = 4
k = 10

# divide dataset into k parts i = 0...k-1
forest_ants['partition'] = random_partitions(len(forest_ants), k, rng)

# initiate vector to hold mean squared errors
e = np.full(k, np.nan)

# for each i
for i in range(k):
#   test dataset = part i
    test_data = forest_ants[forest_ants['partition'] == i]
#   training dataset = remaining data
    train_data = forest_ants[forest_ants['partition'] != i]
#   find f using training dataset
    ortho_poly = Poly(degree=order)
    train_X = ortho_poly.fit_transform(train_data['latitude'])
    train_y = train_data['richness']
    poly_model = LinearRegression()
    poly_model.fit(train_X, train_y)
#   use f to predict for test dataset
    test_X = ortho_poly.transform(test_data['latitude'])
    pred_richness = poly_model.predict(test_X)
#   e_i = prediction error (MSE)
    e[i] = np.mean((test_data['richness'] - pred_richness) ** 2)

# CV_error = mean(e)
cv_error = np.mean(e)
cv_error


# To help us do systematic experiments to explore different combinations
# of `order` and `k` we'll encapsulate the above code as a function.

# Function to perform k-fold CV for a polynomial model on ants data
# forest_ants: dataframe
# k:           number of partitions (scalar, integer)
# order:       degrees of polynomial (scalar, integer)
# return:      CV error as MSE (scalar, numeric)

def cv_poly_ants(forest_ants, k, order):
    forest_ants['partition'] = random_partitions(len(forest_ants), k, rng)
    e = np.full(k, np.nan)
    for i in range(k):
        test_data = forest_ants[forest_ants['partition'] == i]
        train_data = forest_ants[forest_ants['partition'] != i]
    #   train
        ortho_poly = Poly(degree=order)
        train_X = ortho_poly.fit_transform(train_data['latitude'])
        train_y = train_data['richness']
        poly_model = LinearRegression()
        poly_model.fit(train_X, train_y)
    #   test
        test_X = ortho_poly.transform(test_data['latitude'])
        pred_richness = poly_model.predict(test_X)
    #   MSE
        e[i] = np.mean((test_data['richness'] - pred_richness) ** 2)
    cv_error = np.mean(e)
    return(cv_error)


# Test the function
cv_poly_ants(forest_ants, k=10, order=4)
cv_poly_ants(forest_ants, k=22, order=4)


# Next we'll systematically explore a grid of values for k and polynomial order.
# First we'll define a function called `expand_grid()` to make this grid. It
# does the same thing as the `expand.grid()` function in R.

from itertools import product

# Create a data frame with a grid of all combinations of the variables specified
# in grid_dict.
#     https://pandas.pydata.org/pandas-docs/version/0.17.1/
#     cookbook.html#creating-example-data
# grid_dict: the variables to make the grid (dictionary)
# return:    combinations (dataframe)

def expand_grid(grid_dict):
    rows = product(*grid_dict.values())
    return pd.DataFrame.from_records(rows, columns=grid_dict.keys())


# We'll use this function to make a grid of k=5,10,n and order 1 to 8.

grid = expand_grid( {"k": [5,10,len(forest_ants)],
                     "order": range(1,9)})
grid

# Set a random seed so the result is repeatable.

rng = np.random.default_rng(seed=7914)

# Now estimate k-fold CV for each order of the polynomial and the three values
# of k

cv_error = np.full(len(grid), np.nan)
for i in range(len(grid)):
    cv_error[i] = cv_poly_ants(forest_ants, grid["k"][i], grid["order"][i])
cv_error = pd.DataFrame(cv_error, columns=["cv_error"])
result1 = pd.concat([grid, cv_error], axis=1)
result1

# Plot the result.

ggplot(result1) +\
    geom_line(aes(x="order", y="cv_error", color="factor(k)")) +\
    labs(color="k")

# We see that prediction error is very large for order > 7. We need to adjust
# the y-axis limits to zoom in.

ggplot(result1) +\
    geom_line(aes(x="order", y="cv_error", color="factor(k)")) +\
    labs(color="k") +\
    ylim(10,25)

# but now the y limits break the line segments that fall outside the limits. We
# need to use `coord_cartesian()` to set the limits instead.

ggplot(result1) +\
    geom_line(aes(x="order", y="cv_error", color="factor(k)")) +\
    labs(color="k") +\
    coord_cartesian(ylim=(10,25))

# We see that MSE prediction error (cv_error) generally increases for order
# greater than 2 or 3. We also see that cv_error estimates are variable for k=10
# and especially k=5. This is due to the randomness of partitioning a very small
# dataset. If we repeat the above with a different seed, we'd get different
# results for k=5 or k=10. LOOCV is deterministic for this model, so it won't
# differ if we repeat it.
#
# LOOCV (k=22) identifies order=2 as the best performing model, whereas in this
# particular run 10-fold and 5-fold CV identify order=3.
#
# This variability illustrates that we should be mindful that k-fold CV can be
# noisy. What should we do here? Given the uncertainty in MSE estimates for k =
# 5 or 10, we'd be best to use LOOCV as a default (generally a good strategy for
# small datasets). But we could also try for a better estimate by repeated
# k-fold runs. Let's explore the variability in 5-fold and 10-fold CV.

rng = np.random.default_rng(seed=9693) #For reproducible results

grid = expand_grid( {"k": [5,10],
                     "order": range(1,8)})
reps = 100
cv_error = np.full((len(grid), reps), np.nan)
for j in range(reps):
    for i in range(len(grid)):
        cv_error[i,j] = cv_poly_ants(forest_ants, grid["k"][i], grid["order"][i])
        print(j) #monitor progress
cv_error = pd.DataFrame(cv_error, columns=[str(i) for i in range(1, reps+1)])
result2 = pd.concat([grid, cv_error], axis=1)


# Plot the first 10 reps for each k-fold
first10 = result2.iloc[:, :12]
first10['k'] = first10['k'].astype(str) + '-fold CV'
first10_long = first10.melt(id_vars=['k', 'order'], value_vars=[str(i) for i in range(1, 11)],
                            var_name='rep', value_name='cv_error')
first10_long['rep'] = pd.to_numeric(first10_long['rep'])

ggplot(first10_long) +\
    geom_line(aes(x="order", y="cv_error", color="factor(rep)")) +\
    facet_wrap("~k") +\
    coord_cartesian(ylim=(10,25))

# We see again that there is more variability for 5-fold CV. For both 5-fold and
# 10-fold CV there is so much variability, we'd pick different values for order
# on different runs. So, we wouldn't want to rely on a single k-fold run.
# 
# Averaging across runs would give a better estimate of the prediction MSE:

result2['mean_cv'] = result2.iloc[:, 2:].mean(axis=1)

# From the plot of the average for k = 5 and 10, we'd pick the same order as
# LOOCV (k=22).

loocv = result1[(result1['k'] == 22) & (result1['order'] <= 7)]
fold_5_10 = result2[['k', 'order', 'mean_cv']]
fold_5_10 = fold_5_10.rename(columns={'mean_cv': 'cv_error'})
allcombos = pd.concat([fold_5_10, loocv])

ggplot(allcombos) +\
    geom_line(aes(x="order", y="cv_error", color="factor(k)")) +\
    labs(title=f"Mean across {reps} k-fold CV runs", color="k") +\
    coord_cartesian(ylim=(10,25))

# Finally, here is the table of results

allcombos
