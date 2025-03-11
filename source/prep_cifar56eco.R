# Requires keras library to be loaded, which in turn requires tensorflow to be
# installed, and a reticulate Python library to be activated. The function that
# loads the cifar dataset is from the keras Python library.

#' Since this file is large, we'll store it locally in a directory for temporary
#' large datasets and exclude it in `.gitignore` so it doesn't get uploaded to
#' GitHub. It will take a minute or two to download.

prep_cifar56eco <- function() {

    if ( !dir.exists("data_large") ) {
        dir.create("data_large")
        cat("\n# Ignore large data files\ndata_large/", file=".gitignore",
            append=TRUE)
    }
    
    cifar100 <- keras::dataset_cifar100()
    
    # Ecological CIFAR categories and their integer identifiers
    # (cifar_class) and a new integer identifier for these 56 ecological
    # categories
    eco_names <- c("bear", "beaver", "bee", "beetle", "butterfly", "camel", 
              "caterpillar", "cattle", "chimpanzee", "cockroach", "crab", 
              "crocodile", "dinosaur", "dolphin", "elephant", "flatfish", 
              "forest", "fox", "hamster", "kangaroo", "leopard", "lion", 
              "lizard", "lobster", "maple_tree", "mountain", "mouse", 
              "mushroom", "oak_tree", "orchid", "otter", "palm_tree", 
              "pine_tree", "poppy", "porcupine", "possum", "rabbit", 
              "raccoon", "ray", "seal", "shark", "shrew", "skunk", "snail", 
              "snake", "spider", "squirrel", "sunflower", "tiger", "trout", 
              "tulip", "turtle", "whale", "willow_tree", "wolf", "worm")
    cifar_class <- c(3, 4, 6, 7, 14, 15, 18, 19, 21, 24, 26, 27, 29, 30, 31,
                     32, 33, 34, 36, 38, 42, 43, 44, 45, 47, 49, 50, 51, 52, 
                     54, 55, 56, 59, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 
                     77, 78, 79, 80, 82, 88, 91, 92, 93, 95, 96, 97, 99)
    eco_class <- 0:55
    eco_labels <- data.frame(class=eco_class, name=eco_names)

    # Extract and relabel ecological categories
    train_eco <- which(cifar100$train$y %in% cifar_class)    
    test_eco <- which(cifar100$test$y %in% cifar_class)
    x_train <- cifar100$train$x[train_eco,,,]
    x_test <- cifar100$test$x[test_eco,,,]
    y_train <- cifar100$train$y[train_eco,, drop=FALSE]
    y_test <- cifar100$test$y[test_eco,, drop=FALSE]
    
    # Make the new integer response substituting eco_class for cifar_class
    for ( i in 1:nrow(y_train) ) {
        y_train[i,] <- eco_class[cifar_class==y_train[i,]]
    }
    for ( i in 1:nrow(y_test) ) {
        y_test[i,] <- eco_class[cifar_class==y_test[i,]]
    }
    
    save(x_train, x_test, y_train, y_test, eco_labels, 
         file="data_large/cifar56eco.RData")
    
}
