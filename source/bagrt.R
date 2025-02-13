# Bagged regression tree function
# formula:    model formula (formula)
# data:       y and x data (data.frame, mixed)
# xnew_data:  x data to predict from (data.frame, mixed)
# boot_reps:  number of bootstrap replications (scalar, integer)
# return:     bagged predictions (vector, numeric)
# 
bagrt <- function(formula, data, xnew_data, boot_reps=500) {
    n <- nrow(data)
    nx <- nrow(xnew_data)
    boot_preds <- matrix(rep(NA, nx*boot_reps), nrow=nx, ncol=boot_reps)
    for ( i in 1:boot_reps ) {
    #   resample the data (rows) with replacement
        boot_indices <- sample(1:n, n, replace=TRUE)
        boot_data <- data[boot_indices,]
    #   train the base model
        boot_train <- tree(formula, data=boot_data)
    #   record prediction
        boot_preds[,i] <- predict(boot_train, newdata=xnew_data)
    }
    bagged_preds <- rowMeans(boot_preds)
    return(bagged_preds)
}
