### Assignment 4

**Due:** Thursday 10 Apr 11:59 PM

**Grading criteria:** Answer all the questions completely. On time submission.

**Percent of grade:** 7%




#### Neural networks
**Learning goals:**

* Understand the model algorithm of a basic neural network
* Practice using the keras library to fit and tune neural network models
* Practice comparing model accuracy using CV inference algorithms
* Practice visualizing model outputs



#### Part 1: Understanding neural networks

**Q1\.** Code a network by hand. Extend the "by-hand" code in [07_3_ants_neural_net.R](07_3_ants_neural_net.R) to a 2-layer neural network, each layer with 5 nodes. Plot the model predictions. Use these weights and biases (n.b. you don't need to train the model):

```R
w1 <- c(-1.0583408, -0.6441127,  1.1663090,  0.08298533, -0.41105017,
        -0.8540244,  0.5407082, -0.2184951, -0.11781270,  0.36039100,
         0.8608801,  1.2520101, -0.1495921,  0.83325340,  1.15322390,
        -0.5394921, -0.7117111,  0.1879681,  0.30929375,  0.05233159) |>
    matrix(nrow=4, ncol=5, byrow=TRUE)

b1 <- c(0.1909237, 0.5486836, -0.1032256, 0.6253318, 0.2843419)

w2 <- c(0.04039513, 0.7977440,  0.60440171, -0.1800931, -0.210001990,
       -0.14771833, 0.3682977,  0.95937222,  0.3446860,  0.008643006,
       -0.34225080, 1.2922773,  0.11651120,  0.5326685, -0.592227300,
       -0.79168826, 0.5419835, -0.05803596, -1.2168059,  0.169808860,
        0.43390460, 1.0874641,  0.54609700,  0.2390731, -0.599693800) |>
    matrix(nrow=5, ncol=5, byrow=TRUE)

b2 <- c(-0.29183790, 0.32845289, 0.32393071, 0.06806916, -0.01153159)

w3 <- c(-0.3925169, 0.8072395, 1.398517, -0.7064973, -0.3754095) |>
    matrix(nrow=5, ncol=1, byrow=TRUE)

b3 <- 0.3231535
```

Compare plots of predictions for this 2-layer model to the single-layer model. Describe qualitatively (i.e. make a comment) how the predictions differ.

Hint: don't start from scratch, just add a few lines of code here and there where needed. The goal of this is to gain greater understanding of the algorithm.



#### Part 2. Neural networks in practical use

First, get your computer setup with tensorflow and keras. See [07_8_install_keras.md](07_8_install_keras.md) for installing on your local machine (Windows or MacOS), or [08_2_CU_supercomputer_keras_gpu.md](08_2_CU_supercomputer_keras_gpu.md) for getting set up on the supercomputer.

**Q2\.** Using keras, train and tune a neural network to predict the occurrence of plant species "nz05" from the previous assignment.

Suggestions:

* Use a feedforward network
* Use binary_crossentropy loss
* Calculate the F1 score, AUC, and overall error rate
* Tuning: we don't have time to try many combinations or to do k-fold CV (you would do this for a research project). Here is a strategy:
  * use the cross validation option built into keras (i.e. `fit()` argument `validation_split=0.2`)
  * try these four different architectures: 25 wide, 50 wide, 5x5 deep, 5x10 deep
  * try models with and without dropout regularization applied to the layers, with 0.3 as a default rate (see example from class) 
* Early stopping: usually we stop learning after some number of epochs to prevent overfitting (i.e. when you see the validation error start to go back up). How many epochs are needed?
* Compare the models using the F1 metric. Which is best? Speculate on why this model might be best.

**Q3\.** Compare to the random forest and boosting models.

* How does your neural network perform compared to the other models? Does it get within the same ballpark?

**Q4\.** Plot probability and occurrence predictions as you did for the boosting model.

* Comment on any similarities or differences in the prediction of the neural network compared to the boosting model.

