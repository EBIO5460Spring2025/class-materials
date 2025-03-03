# Install xgboost from conda-forge
# https://xgboost.readthedocs.io/en/stable/install.html
# for CU supercomputer, install the gpu enabled version

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xgboost as xgb
from plotnine import *

# Load presence-absence data for species "nz05"
nz05df = pd.read_csv("data/nz05.csv")
nz05df.head()

# Outline of New Zealand (header warning is harmless)
nzpoly = gpd.read_file("data/nzpoly.geojson")
type(nzpoly)

# Plot the records
nz05dfs = nz05df.sort_values("occ") #present on top
fig, ax = plt.subplots()
nzpoly.plot(ax=ax, color="lightgray")
scatter = ax.scatter(nz05dfs['x'], nz05dfs['y'], s=0.5, c=nz05dfs['occ'], 
                     alpha=0.2)
legend1 = ax.legend(*scatter.legend_elements(), title="occ")
ax.add_artist(legend1)
plt.show()

# Data for modeling
nz05pa = nz05df.drop(columns=["group","siteid","x","y","toxicats"])
nz05pa.head()

# -- Example boosted model (5 secs)

# Prepare data for xgboost
nz05pa_xgb = xgb.DMatrix(data=nz05pa.drop(columns=["occ"]), label=nz05pa["occ"])
type(nz05pa_xgb)
nz05pa_xgb.feature_names

# Train
param = {"max_depth": 1, "eta": 0.01, "nthread": 2, "objective": "binary:logistic"}
nz05_train = xgb.train(param, nz05pa_xgb, num_boost_round=10000) 

# Predict
nz05_prob = nz05_train.predict(nz05pa_xgb)
nz05_pred = 1 * (nz05_prob > 0.5)
type(nz05_pred)

# Characteristics of this prediction
plt.figure()
plt.hist(nz05_prob)
plt.show()
np.max(nz05_prob)
np.sum(nz05_prob > 0.5) #number of predicted presences

pd.crosstab(nz05_pred, nz05pa["occ"])  #confusion matrix
np.mean(nz05_pred == nz05pa["occ"]) #accuracy
np.mean(nz05_pred != nz05pa["occ"]) #error = 1 - accuracy

# -- Example prediction for a grid of the predictor variables across NZ

# Read in the grid of predictor variables
NZ_grid = pd.read_csv("data/NZ_predictors.csv")
NZ_grid.head()

# Prepare data for xgboost
NZ_grid_xgb = xgb.DMatrix(data=NZ_grid.drop(columns=["x","y"]))
NZ_grid_xgb.feature_names #matches the training data

#Predict
pred = NZ_grid[["x","y"]].copy()
pred["prob"] = nz05_train.predict(NZ_grid_xgb)
pred["present"] = 1 * (pred["prob"] > 0.5)

# Map probability prediction
plt.close()
(ggplot()
+ geom_raster(aes(x="x", y="y", fill="prob"), data=pred)
+ scale_fill_gradientn(colors=plt.cm.viridis.colors)
+ coord_equal()
+ theme_void()
+ labs(fill = "Probability"))

# Map presence prediction (not optimal but this works)
# Things that don't work: geom_tile; fill="factor(present)"
(ggplot()
+ geom_raster(aes(x="x", y="y", fill="present"), data=pred)
+ coord_equal()
+ theme_void()
+ labs(fill = "Present"))
