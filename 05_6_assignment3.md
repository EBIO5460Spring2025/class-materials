### Assignment 3

**Due:** Tuesday 25 Feb 11:59 PM

**Grading criteria:** Answer all the questions completely. On time submission.

**Percent of grade:** 7%

**Format for submitting assignments:**

* Submit code and answers to questions as comments in the same script or markdown report.
* All code and text in one file please (i.e. not a separate file per question).
* Please include all the produced plots along with your code. I don't want to run your code to make plots!! You may produce a report if you wish (e.g. markdown, html, pdf) but a script file plus plots is sufficient.
* Please organize your repository with files clearly labelled to indicate belonging to the assignment, or put each assignment in its own directory (folder).



#### Classification trees and bagging

**Learning goals:**

* Understand the steps of the bagging model algorithm
* Practice modifying a function to examine its internal workings
* Understand the differences between regression and classification algorithms
* Practice tuning machine learning algorithms using the CV inference algorithm
* Practice comparing model accuracy using the CV inference algorithm
* Understand how the bagging algorithm averages over probabilities in classification models
* Practice visualizing model outputs



The goal of this assignment is to compare the predictive performance (using k-fold CV) of a decision-tree model and a bagged decision tree model to the KNN model from the previous assignment for the presence/absence of the Willow Tit. We will again use the single predictor, elevation.

**Q1\.** Train a classification tree. For R, use the `tree()` function from the `tree` package. The option `type="class" ` in `predict()` will return the predicted class (see `?predict.tree`). Alternatively, for Python, use the `tree.DecisionTreeClassifier()` function from the [scikit-learn](https://scikit-learn.org) package (see, e.g. in `04_5_ants_bag.py`). Tune the tree for the minimum number of samples to split a node (R: `mincut`; Python: `min_samples_split`) using k-fold cross validation. Feel free to share your tuning plot on Piazza and see what others are getting.

**Q2\.** Plot predictions (as a map) from the classification tree. The code for this will be substantially similar to what you did in the previous assignment.

**Q3\.** Modify the `bagrt()` function to make a `bagct()` function for bagged classification trees. This is very similar to the step we made from regression to classification for the KNN model (look at `03_5_classification_knn.R/py` for inspiration). Whereas for bagged regression trees we averaged the prediction for a numerical response variable (e.g. richness in the ants example) across trees, for bagged classification trees we need to average the probabilities across trees before making the final present/absent prediction (equivalently we can vote across trees).  For R, the option `type="vector" ` in `predict()` will return the probabilities (see `?predict.tree`). For Python, the function `tree.predict_proba()` will return the probabilities (see [sklearn API](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict)). 

**Q4\.** Tune `bagct()` for the number of bootstrap samples on the Willow Tit data. Yep, you know the gig by now: use k-fold CV. Feel free to share your tuning plot on Piazza and see what others are getting.

**Q5\.** Plot predictions (as a map) from the tuned, bagged classification tree algorithm.

**Q6\.** Compare predictive accuracy of KNN, simple tree, and bagged tree models for the Willow Tit dataset by collating the CV results. Which model is best?

**Q7\.** Modify the code for your **best-performing model** to output the probabilities instead of presence/absence class. Plot the probabilities as a map.

