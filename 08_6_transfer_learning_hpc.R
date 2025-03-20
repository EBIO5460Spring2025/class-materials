# First log in to research computing
# Transfer to GPU node
#   sinteractive --partition=atesting_a100 --time=0:60:00 --nodes=1 --ntasks=8 --gres=gpu:1
# Change to project directory
#   cd /projects/<username>/ml4e
# Start conda
#   module load anaconda
# Activate conda environment
#   conda activate r-tf2150py3118
# Start R
#   R

# Script pared down for use on an HPC resource

reticulate::use_condaenv(condaenv = "r-tf2150py3118")
tensorflow::tf_gpu_configured(verbose = TRUE) #check GPU, status TRUE is good
library(keras)

# Load and prepare data
load("data_large/cifar56eco.RData")
x_train_inet <- imagenet_preprocess_input(x_train)
x_test_inet <- imagenet_preprocess_input(x_test)
y_train <- to_categorical(y_train, 56)
rm(x_train, x_test)

# Specify the model
vgg16_base <- application_vgg16(weights="imagenet", include_top=FALSE, 
                                input_shape=c(224, 224, 3))
freeze_weights(vgg16_base)
modtfr1 <- keras_model_sequential(input_shape=c(32, 32, 3)) |>
    layer_resizing(224, 224) |>
    vgg16_base() |>
#   Flatten with dropout regularization
    layer_flatten() |>
    layer_dropout(rate=0.5) |>
#   Standard dense layer
    layer_dense(units=512) |>
    layer_activation_relu() |>
#   Output layer (56 categories)    
    layer_dense(units=56) |> 
    layer_activation_softmax()

# Compile, train, save
compile(modtfr1, loss="categorical_crossentropy", optimizer="rmsprop",
        metrics="accuracy")
fit(modtfr1, x_train_inet, y_train, epochs=30, batch_size=128, 
    validation_split=0.2) -> history
save_model_tf(modtfr1, "saved/modtfr1")
save(history, file="saved/modtfr1_history.Rdata")

# Test set prediction
pred_prob <- predict(modtfr1, x_test_inet)
save(pred_prob, file="saved/modtfr1_pred_prob.Rdata")
pred_cat <- as.numeric(k_argmax(pred_prob))
mean(pred_cat == drop(y_test)) #accuracy on test set
