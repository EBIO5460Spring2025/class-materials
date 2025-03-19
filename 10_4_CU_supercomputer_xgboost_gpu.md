## Install R xgboost for GPU

Login, then ask for an interactive job on the `atesting_a100` partition.

```bash
sinteractive --partition=atesting_a100 --time=0:30:00 --nodes=1 --ntasks=1 --gres=gpu:1
```

It's often nice to clear the console once you're transferred to the testing node:

```bash
clear
```

Install R  and zlib into a new conda environment (zlib is a software library needed as a dependency for the data.table package in R)

```bash
module load anaconda
conda create --name r-xgboost-gpu
conda activate r-xgboost-gpu
conda install r zlib -c conda-forge
```

Find the url for the latest GPU version of xgboost. I found it under "Experimental binary packages for R with CUDA enabled" here:

https://github.com/dmlc/xgboost/releases

CUDA is the code library for GPU workflows on NVIDIA GPU hardware. We are using an [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/).



Start R, then install dependencies needed for xgboost:

```R
install.packages("data.table")
install.packages("jsonlite")
```

Install xgboost using the url found above:

```R
xgb_url <- "https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/release_3.0.0/aae009b9277d7a981ca59aa2c5227b09c44545a2/xgboost_r_gpu_linux.tar.gz"
install.packages(xgb_url, repos=NULL, type="source")
```

If this is successful, that's it for installation. Now, staying in the current R session, test xgboost on GPU.

```R
library(xgboost)

# Generate a synthetic dataset

set.seed(123)
n <- 10000  # Number of samples
p <- 20     # Number of features
data <- matrix(rnorm(n * p), nrow = n, ncol = p)
labels <- rbinom(n, 1, 0.5)  # Binary target variable

# Create DMatrix for XGBoost

dtrain <- xgb.DMatrix(data = data, label = labels)

# Set parameters for XGBoost

params <- list(
  objective = "binary:logistic",
  max_depth = 6,
  eta = 0.3,
  eval_metric = "logloss",
  tree_method="hist",        #default but needed for GPU
  device = "cuda"            #critical: tells xgboost to use GPU
)

# Train the XGBoost model

system.time(
bst <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 10000,  # Number of boosting rounds
  verbose = 1
)
)
```

If everything is working, it should be quicker than training on CPU. On CPU I found that the quickest times were with 16-32 cores and bottomed out around 20 seconds. On GPU it was about 2X faster:

   user  system elapsed
  9.600   0.045   9.682

Once you're done, quit R by typing `q()`; don't save the workspace. Then from the shell prompt, deactivate the conda environment

````
conda deactivate
````

and `exit` from the interactive node back to the login node. You'll probably want to `clear` the console again.
