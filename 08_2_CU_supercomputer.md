# CU supercomputer

We have free access to amazing computing resources for machine learning. CU has a supercomputer with NVIDIA 80GB A100 tensor core GPU nodes. These are specialized GPUs for machine learning. These are roughly \$20K each. Each compute node on CUs Alpine supercomputing cluster has three of these, so \$60K worth of compute. You should use them!!



## Sign up

Signing up for a Research Computing account only takes a few minutes and you will get approval almost right away. It may ask for a reason. Put "EBIO 5460 Machine Learning for Ecology class, spring 2024". To sign up, go [here](https://www.colorado.edu/rc/) and click "Request an Account". You also need to have set up [Duo 2 factor authentication](https://oit.colorado.edu/services/identity-access-management/duo-multi-factor-authentication#useduo) for your CU Boulder identikey.



## Login to computing resources

You can SSH from a terminal

```bash
ssh jbsmith@login.rc.colorado.edu
```

Instead of using a terminal and typing `ssh` commands, you can also use the opensource standalone terminal program Putty, which has the advantage of storing all your connection setups.

For university servers you may need to [VPN to the campus network first](https://oit.colorado.edu/services/network-internet-services/vpn).

You will be prompted for username and password. Once you enter your password, it will seem that nothing is happening. You now must open the DUO app on your phone and approve the push notification. Then you'll get a prompt that looks something like this:

```bash
You are using login node: login12
[jbsmith@login12 ~]$
```



## General Linux and R commands

For managing files using linux commands, GUI applications for transfering files, and for handy R commands on a server, see [05_3_R_on_server.md](05_3_R_on_server.md).



## Install tool chain for R keras on Alpine GPU

Request transfer to a GPU compute node. Give yourself an hour, as conda might need it!

```bash
sinteractive --partition=atesting_a100 --time=0:60:00 --nodes=1 --ntasks=8 --gres=gpu:1
```

Your prompt will change to something like:

```
[jbsmith@c3gpu-c2-u13 ~]$
```

Start the Anaconda module

```bash
module load anaconda
```

Your prompt will change to something like:

```
(base) [jbsmith@c3gpu-c2-u13 ~]$
```

indicating that you are in the base conda environment. We now want to set up an environment specifically for R, keras and tensorflow. What we are doing is installing all of the necessary software into an isolated computing environment. First create a new conda environment:

```bash
conda create --name r-tf2150py3118
```

Now activate that environment

```bash
conda activate r-tf2150py3118
```

You should see the environment change in your prompt

```
(r-tf2150py3118) [jbsmith@c3gpu-c2-u13 ~]$
```

Now install all the needed software. This may take as long as 20 minutes.

```bash
conda install r python=3.11 tensorflow-gpu=2.15.0 tensorflow-hub tensorflow-datasets scipy requests Pillow h5py pandas pydot -c conda-forge
```



## Using R keras

It is generally best to work from your projects directory on RC computing because this has much more storage for large files, which you'll likely need for machine learning projects. We don't need to do that just yet but here is how you get there;

```bash
cd /projects/jbsmith
```

You can the set up directories here for specific projects. For example, let's make a directory for this class, and navigate to it:

```
mkdir ml4e
cd ml4e
```



Start R by typing `R` at the prompt. Install the `keras` library.

```
install.packages("keras")
```

You will need to install any other R packages you want to use, such as `ggplot2` or `dplyr`, the same way.

Now we can use the R keras library. Work through this example to check that everything is working. This is the canonical keras example.

```r
# Tell reticulate which conda environment to use
reticulate::use_condaenv(condaenv = "r-tf2150py3118")

# Check tensorflow GPU configuration. Should list and name all 
# GPU devices with status TRUE
tensorflow::tf_gpu_configured(verbose = TRUE)

# Now you can load keras (must be after `use_condaenv` above)
library(keras)

# Mnist handwritten letters dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# recode
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Define, compile and fit the model (2 layer feedforward network)
model <- keras_model_sequential(input_shape = c(784)) |>
  layer_dense(units = 256, activation = 'relu') |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 128, activation = 'relu') |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 10, activation = 'softmax')

compile(model, loss='categorical_crossentropy', optimizer=optimizer_rmsprop(),
        metrics=c('accuracy'))

fit(model, x_train, y_train, epochs=30, batch_size=128, validation_split=0.2)

# Fit should use GPU. It will take a few moments to set up the GPU the first time.

get_weights(model)
```

When you have finished with R, type `q()` to quit R.  To end your session on the compute node, type `exit`. This will bring you back to the login node. You might want to type `clear` to clear the screen as the prompt will often jump to somewhere randomly in the printed output. To logout, type `logout`.