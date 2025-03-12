# First log in to research computing
# Transfer to GPU node
#   sinteractive --partition=atesting_a100 --time=0:60:00 --nodes=1 --ntasks=8 --gres=gpu:1
# Change to project directory
#   cd /projects/<username>/ml4e
# Start conda
#   module load anaconda
# Activate conda environment
#   r-tf2150py3118
# Start R
#   R

# Script pared down for use on an HPC resource

# A small hitch is that we can't set a seed for use on GPU, and it's important
# that we don't. If we set a seed we'll be stuck on CPU. This means our exact
# results are not reproducible. Unfortunately this isn't a solved problem.

reticulate::use_condaenv(condaenv = "r-tf2150py3118")
tensorflow::tf_gpu_configured(verbose = TRUE) #check GPU, status TRUE is good
library(keras)

# Load and prepare data

# library(dplyr)
source("source/prep_cifar56eco.R")
if ( !file.exists("data_large/cifar56eco.RData") ) {
    prep_cifar56eco()
}
load("data_large/cifar56eco.RData")
x_train <- x_train / 255 #convert image data to 0-1 scale
x_test <- x_test / 255
y_train_int <- y_train #copy of integer version for labelling later
y_train <- to_categorical(y_train, 56) #convert integer response to dummy

# Specify the model
modcnn1 <- keras_model_sequential(input_shape=c(32,32,3)) |>
#   1st convolution-pool layer sequence
    layer_conv_2d(filters=32, kernel_size=c(3,3), padding="same") |>
    layer_activation_relu() |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
#   2nd convolution-pool layer sequence    
    layer_conv_2d(filters=64, kernel_size=c(3,3), padding="same") |> 
    layer_activation_relu() |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
#   3rd convolution-pool layer sequence    
    layer_conv_2d(filters=128, kernel_size=c(3,3), padding="same") |> 
    layer_activation_relu() |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
#   4th convolution-pool layer sequence
    layer_conv_2d(filters=256, kernel_size=c(3,3), padding="same") |> 
    layer_activation_relu() |> 
    layer_max_pooling_2d(pool_size=c(2,2)) |>
#   Flatten with dropout regularization
    layer_flatten() |>
    layer_dropout(rate=0.5) |>
#   Standard dense layer
    layer_dense(units=512) |>
    layer_activation_relu() |>
#   Output layer with softmax (56 categories to predict)    
    layer_dense(units=56) |> 
    layer_activation_softmax()

# Compile, train, save
compile(modcnn1, loss="categorical_crossentropy", optimizer="rmsprop",
        metrics="accuracy")
fit(modcnn1, x_train, y_train, epochs=30, batch_size=128, 
    validation_split=0.2) -> history
save_model_tf(modcnn1, "saved/modcnn1")
save(history, file="saved/modcnn1_history.Rdata")

# Test set prediction
pred_prob <- predict(modcnn1, x_test)
save(pred_prob, file="saved/modcnn1_pred_prob.Rdata")
pred_cat <- as.numeric(k_argmax(pred_prob))
mean(pred_cat == drop(y_test))
