# Spam-Filter
Spam filtering using Naive Bayes

* requirements
* Python version 3.10 or higher
* Jupyter Notebook

1. Load the Libraries, pandas, numpy, math

2. Read in the spambase data

3. Seperate the train and test data

4. Change the hyper parameter,
  k = a variation of laplase smoothing 
  higher K, the better the accuracy up to ~ 89%
  K = 0 has no change in laplase smoothing

5. Run Naive_Bayes(test, train, true_test_class,true_train_class, K)

6. Run the Confusion matrix to see the predicted vs actual spam/non-spam results
