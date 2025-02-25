# Ant data: neural network
# Brett Melbourne
# 27 Feb 2024

# A single layer neural network, or feedforward network, illustrated with the
# ants data. We first hand code the model algorithm as a proof of understanding.
# Then we code the same model and train it using Keras.

import numpy as np
import pandas as pd
from plotnine import *
from source.ml4e import expand_grid

# Ant data with 3 predictors of species richness
ants = pd.read_csv("data/ants.csv")
ants = ants[["richness", "latitude", "habitat", "elevation"]]
ants.head()

# Scaling parameters
lat_mn = np.mean(ants["latitude"])
lat_sd = np.std(ants["latitude"], ddof=1) #sample sd, same as R code
ele_mn = np.mean(ants["elevation"])
ele_sd = np.std(ants["elevation"], ddof=1)


# Hand-coded feedforward network

# Before we go on to Keras, we're going to hand code a feedforward network in
# Python to gain a good understanding of the model algorithm. Here is our
# pseudocode from the lecture notes.

# Single layer neural network, model algorithm:
#
# define g(z)
# load X, i=1...n, j=1...p (in appropriate form)
# set K
# set weights and biases: w(1)_jk, b(1)_k, w(2)_k1, b(2)_1
# for each activation unit k in 1:K
#     calculate linear predictor: z_k = b(1)_k + Xw(1)_k
#     calculate nonlinear activation: A_k = g(z_k)
# calculate linear model: f(X) = b(2)_1 + Aw(2)_1
# return f(X)


# Now code this algorithm in Python. I already trained this model using Keras
# (see later) to obtain a parameter set for the weights and biases.

# Single layer neural network, model algorithm

# define g(z)
def g_relu(z):
    g_z = np.where(z < 0, 0, z)
    return g_z

# load x (could be a grid of new predictor values or the original data)
grid_data = expand_grid(
    {"latitude": np.linspace(np.min(ants["latitude"]), np.max(ants["latitude"]), 201),
     "habitat":  ["forest","bog"],
     "elevation": np.linspace(np.min(ants["elevation"]), np.max(ants["elevation"]), 51)}
)

# data preparation: scale, dummy encoding, convert to matrix
x = grid_data.copy()
x["latitude"] = (x["latitude"] - lat_mn) / lat_sd
x["elevation"] = (x["elevation"] - ele_mn) / ele_sd
x["bog"] = np.where(x["habitat"] == "bog", 1, 0)
x["forest"] = np.where(x["habitat"] == "forest", 1, 0)
x = x[["latitude", "bog", "forest", "elevation"]]
x = x.to_numpy()

# dimensions of x
n = x.shape[0] #rows
p = x.shape[1]

# set K
K = 5

# set parameters (weights and biases)
w1 = np.array([[-0.2514450848, 0.4609818,  0.1607399, -0.9136779, -1.0339828],
               [-0.4243144095, 0.7681985, -0.1529205, -0.3439012,  0.8026423],
               [-0.0005548226, 0.6407318,  1.4387618,  1.6372939,  1.1695395],
               [-0.0395327508, 0.5222837, -0.6239772, -0.3365386, -0.7156096]])

b1 = np.array([-0.2956778, 0.3149067, 0.8800480, 0.6910487, 0.6947369])
b1 = [-0.2956778, 0.3149067, 0.8800480, 0.6910487, 0.6947369]

w2 = np.array([[-0.4076283],
               [ 0.6379358],
               [ 0.8768858],
               [ 1.6320601],
               [ 0.9864114]])

b2 = 0.8487959


# hidden layer 1, iterating over each activation unit
A = np.full((n,K), np.nan)
for k in range(K):
#   linear predictor (dot is matrix multiplication in numpy)
    z = x.dot(w1[:,k]) + b1[k]
#   nonlinear activation
    A[:,k] = g_relu(z)

# output, layer 2, linear model
f_x = A.dot(w2) + b2

# return f(x)
f_x


# Plot predictions
nn1_preds = pd.DataFrame(f_x, columns=["richness"])
preds = pd.concat([grid_data, nn1_preds], axis=1)

(ggplot()
+ geom_line(aes(x="latitude", y="richness", color="elevation",
                group="factor(elevation)"),
                data=preds, linetype="dashed")
+ geom_point(aes(x="latitude", y="richness", color="elevation"),
                data=ants)
+ facet_wrap("~habitat")
+ theme_bw())


# Using Keras to fit neural networks

# You'll need to install Python and Tensorflow, which will also install the
# keras Python library. The keras library is an interface to the Python
# Tensorflow library, which in turn is an interface to Tensorflow (mostly C++)!
# See Assignment 4 for installation directions into a conda environment.

import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

# Set random seed for reproducibility
keras.utils.set_random_seed(5574)

# Prepare data
xtrain = ants.copy()
xtrain["latitude"] = (xtrain["latitude"] - lat_mn) / lat_sd
xtrain["elevation"] = (xtrain["elevation"] - ele_mn) / ele_sd
xtrain["bog"] = np.where(xtrain["habitat"] == "bog", 1, 0)
xtrain["forest"] = np.where(xtrain["habitat"] == "forest", 1, 0)
xtrain = xtrain[["latitude", "bog", "forest", "elevation"]]
xtrain = xtrain.to_numpy()

ytrain = ants["richness"]

# Specify the model
modnn1 = keras.Sequential()
modnn1.add(keras.Input(shape=xtrain.shape[1]))
modnn1.add(layers.Dense(units=5))        
modnn1.add(layers.Activation("relu"))
modnn1.add(layers.Dense(units=1))

# Check configuration
modnn1.summary()

# Compile and train the model
modnn1.compile(optimizer="rmsprop", loss="mse")
history = modnn1.fit(xtrain, ytrain, epochs=300, batch_size=4)
history = pd.DataFrame(history.history)

# To visualize model fit in real time, you can use tensorboard
# https://keras.io/api/callbacks/tensorboard/
# https://www.tensorflow.org/tensorboard/get_started

# Save model and history, or load it back in
# tf.saved_model.save(modnn1, "07_3_ants_neural_net_files/saved/modnn1")
# history.to_csv("07_3_ants_neural_net_files/saved/modnn1_history.csv", index=False)
# modnn1 = keras.models.load_model("07_3_ants_neural_net_files/saved/modnn1")
# history = pd.read_csv("07_3_ants_neural_net_files/saved/modnn1_history.csv")

# Plot history
history.plot()
plt.show()

# Make predictions for predictor grid
npred = modnn1.predict(x)
npred = pd.DataFrame(npred, columns=["richness"])
preds = pd.concat([grid_data, npred], axis=1)

(ggplot()
+ geom_line(aes(x="latitude", y="richness", color="elevation",
                group="factor(elevation)"),
                data=preds, linetype="dashed")
+ geom_point(aes(x="latitude", y="richness", color="elevation"),
                data=ants)
+ facet_wrap("~habitat")
+ theme_bw())    

# Get weights and biases
modnn1.get_weights()
