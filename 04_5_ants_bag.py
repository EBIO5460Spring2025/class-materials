import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib.pyplot import show
from plotnine import *


# Start random number generator
rng = np.random.default_rng()

# Forest ant data:
ants = pd.read_csv("data/ants.csv")
forest_ants = ants[ants["habitat"]=="forest"]
forest_ants = ants[["latitude","richness"]]
forest_ants.head()


# Train a single decision tree model for comparison
dt = tree.DecisionTreeRegressor(max_depth=2)
tree_trained = dt.fit(forest_ants[["latitude"]], forest_ants["richness"])
tree.plot_tree(tree_trained)
show()

# Grid of latitudes to predict for
grid_data = pd.DataFrame(np.linspace(np.min(forest_ants["latitude"]),
                                     np.max(forest_ants["latitude"]), 201),
                         columns=["latitude"])

# Predictions for the single tree model
preds_1tree = tree_trained.predict(grid_data)
preds_1tree = pd.DataFrame(preds_1tree, columns=["richness"])   
preds_1tree = pd.concat([grid_data, preds_1tree], axis=1)                     

(ggplot(forest_ants, aes(x="latitude", y="richness"))
+ geom_point()
+ geom_line(data=preds_1tree)
+ coord_cartesian(ylim=(0,20))
+ labs(title="Single regression tree"))


# Bagging algorithm
boot_reps = 500
dt = tree.DecisionTreeRegressor(max_depth=2) #define base model
n = len(forest_ants)
nx = len(grid_data)
boot_preds = np.full((nx, boot_reps), np.nan)
for i in range(boot_reps):
    # resample the data (rows) with replacement
    boot_indices = rng.choice(range(n), n, replace=True)
    boot_data = forest_ants.iloc[boot_indices]
    # train the base model
    boot_train = dt.fit(boot_data[["latitude"]], boot_data["richness"])
    # record prediction
    boot_preds[:,i] = boot_train.predict(grid_data)
# mean of predictions
bagged_preds = np.mean(boot_preds, axis=1)


# Predictions for the bagged tree model
preds_bag = pd.DataFrame(bagged_preds, columns=["richness"])
preds_bag = pd.concat([grid_data, preds_bag], axis=1)

(ggplot(forest_ants, aes(x="latitude", y="richness"))
+ geom_point()
+ geom_line(data=preds_1tree)
+ geom_line(data=preds_bag, color="blue")
+ coord_cartesian(ylim=(0,20))
+ labs(title="Bagged regression tree (blue) vs single regression tree (black)"))
