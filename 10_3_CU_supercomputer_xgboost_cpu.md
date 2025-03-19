## Install R xgboost for CPU

Login, then ask for an interactive job on the `atesting` partition.

```bash
sinteractive --partition=atesting --ntasks-per-node=8 --nodes=1 --time=00:30:00
```

It's often nice to clear the console once you're transferred to the testing node:

```bash
clear
```

Install R into a new conda environment

```bash
module load anaconda
conda create --name r-xgboost-cpu
conda activate r-xgboost-cpu
conda install r -c conda-forge
```

Start R, then

```R
install.packages("xgboost")
install.packages("future")
```

That's it for installation. Now, staying in the current R session, test xgboost on CPU, especially to see if its parallel implementation is working as expected.

```R
library(xgboost)
library(future) #for availableCores()

# Generate a synthetic dataset
set.seed(123)
n <- 10000  # Number of samples
p <- 20     # Number of features
data <- matrix(rnorm(n * p), nrow = n, ncol = p)
labels <- rbinom(n, 1, 0.5)  # Binary target variable

# Create DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = data, label = labels)

# Get the number of available CPU cores
availableCores() #number of cores per node
availableWorkers() #list of workers that have been allocated
num_cores <- length(availableWorkers())
num_cores  #Is it what you expected? It should be what you asked for.

# Set parameters for XGBoost, using all available cores
params <- list(
  objective = "binary:logistic",
  max_depth = 6,
  eta = 0.3,
  eval_metric = "logloss",
  nthread = num_cores  # Use all available CPU cores
)

# Train the XGBoost model
system.time(
bst <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 10000,
  verbose = 1
)
)

# Now set nthread to 1 and time again
params <- list(
  objective = "binary:logistic",
  max_depth = 6,
  eta = 0.3,
  eval_metric = "logloss",
  nthread = 1  # Use only 1 CPU core
)

system.time(
bst <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 10000,
  verbose = 1
)
)
```


If everything is working, it should be much quicker to train on 8 cores than on 1 core. We want to look at the elapsed time. Typical output:

8 core

   user  system elapsed
202.544   0.247  25.513

1 core

   user  system elapsed
141.310   0.000 141.666

In this example, 8 cores provides a 5X speed up. Now quit R by typing `q()`; don't save the workspace. Then from the shell prompt, deactivate the conda environment

````
conda deactivate
````

 and `exit` from the interactive node back to the login node. You'll probably want to `clear` the console again.



### Test as a batch job with more cores

Copy the following R code into a file called `testxgboost.R`. This is the same as the multicore part of the code above but we've added a new line at the top, `options(echo = TRUE)`, which tells R to print each line of code to the output stream. You can omit this line, then only the lines that direct R to print (including implicit print) will be printed to the output stream.

```R
options(echo = TRUE)
library(xgboost)
library(future) #for availableCores()

# Generate a synthetic dataset
set.seed(123)
n <- 10000  # Number of samples
p <- 20     # Number of features
data <- matrix(rnorm(n * p), nrow = n, ncol = p)
labels <- rbinom(n, 1, 0.5)  # Binary target variable

# Create DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = data, label = labels)

# Get the number of available CPU cores
availableCores() #number of cores per node
availableWorkers() #list of workers that have been allocated
num_cores <- length(availableWorkers())
num_cores  #is it what you expected? It should be what you asked for.

# Set parameters for XGBoost
params <- list(
  objective = "binary:logistic",
  max_depth = 6,
  eta = 0.3,
  eval_metric = "logloss",
  nthread = num_cores  # Use all available CPU cores
)

# Train the XGBoost model
system.time(
bst <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 10000,
  verbose = 1
)
)
```

To do this you could make the file on your computer and then use a file transfer tool to get it across to the supercomputer. Alternatively, you could quickly use the nano text editor on the supercomputer:

```bash
nano textxgboost.R
```

Copy the code above, then paste into nano (right-click to paste).

Exit nano, saving the file:
```
ctrl-x
# choose yes to save buffer when prompted
# press enter to save the file as default filename
```

Check the file

```
cat testxgboost.R
```

Now set up the job batch file (`.sh` indicates a shell script):

```bash
nano testxgboost.sh
```

Copy the following into nano. We are asking for a job on `amilan` , the main partition of the CU supercomputer. Each node of amilan has 64 cores. We'll ask for 1 node with 16 cores. This job won't take long, so we'll ask for a maximum time of 10 minutes. Enter your email address. The lines with `echo` will print to the output file along with any text sent to the output stream. The line that runs the R code is `Rscript --vanilla testxgboost.R`.

```bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16 #number of cores per node (max 64)
#SBATCH --time=00:10:00 # hours:minutes:seconds
#SBATCH --partition=amilan
#SBATCH --mail-type=ALL
#SBATCH --mail-user=put_your_email_address_here
#SBATCH --output=xgboost_test_%j.out #output file (%j adds job id)

echo "== Starting Job =="

echo "== Loading conda module =="
module purge
module load anaconda

echo "== Activating conda environment =="
conda activate r-xgboost-cpu

echo "== Starting R =="
Rscript --vanilla testxgboost.R

echo "== End of Job =="
```

Exit nano, saving the file, as we did above.

Check the file

```
cat testxgboost.sh
```

Now we can submit the job to the slurm system, which will put it in the queue.

```
sbatch testxgboost.sh
```

When the job starts and again when it finishes, you'll be sent an email. We can also check on its status in the queue:

```bash
squeue --user=put_your_user_name_here --start
```

See explanatory codes for squeue [here](https://curc.readthedocs.io/en/latest/running-jobs/squeue-status-codes.html).

When the job is finished, the output will be in the file named `xgboost_test_<job_number>.out`, where the job number is appended to the file name by slurm. Type `ls` to see the current files in the working directory, and type:

```bash
cat xgboost_test_<job_number>.out
```

to see the output. Tip: instead of typing the above, it's quicker to copy/paste from the `ls` output by selecting the file name (double left click), which will automatically copy the text, then right click to paste (you don't need to move the cursor).

You can now experiment with different numbers of cores by changing the batch script and running another job:

```bash
nano testxgboost.sh
sbatch testxgboost.sh
```
