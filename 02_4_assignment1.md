### Assignment 1

**Due:** Thursday 1 Feb 11:59 PM

**Grading criteria:** Complete all the check boxes. On time submission.

**Percent of grade:** 7%



#### Coding the LOOCV algorithm

**Learning goals:** 

* Understand the steps of the cross validation algorithm
* Build competence in translating algorithms to code
* Practice tuning a machine learning algorithm using the CV inference algorithm

Coding algorithms gives you much deeper understanding for how they work, provides detailed knowledge of what they are actually doing, and builds intuition that you can draw on throughout your career.

We can use the CV algorithm that we coded in class to do LOOCV. But LOOCV is a special case that suggests an even simpler algorithm. Code up the LOOCV algorithm in R or Python from the following pseudocode.

```
# LOOCV algorithm
# for each data point
#     fit model without point
#     predict for that point
#     measure prediction error (compare to observed)
# CV_error = mean error across points
```

Use the first section of [02_2_ants_cv_polynomial.R](02_2_ants_cv_polynomial.R) to get going with reading in the data and using a polynomial model.

- [ ] As we did for coding the k-fold CV algorithm, first code the LOOCV algorithm line by line.

- [ ] Then turn it into a function.

- [ ] Finally, use the function to investigate the LOOCV error for different orders of the polynomial model to determine the order with the best predictive accuracy.



**Push your code to GitHub**

