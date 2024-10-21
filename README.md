# learn-ML-DL-algorithm

## 01 - Multivariate linear regression

Applied multivariate linear regression to predict house price based on area and number of bedrooms.

- **standardize_train_set** function : used for standardize features in training set.
- **standardize_test_set** fucntion : used for standardize features in testing set.
- **linearRegression class:** used to build a model based on linear regression. This class contains :  
  _ *calculate_absolute_error*  
  _ _gradient function_  
  _ *gradient_descent function*  
  _ _predict funtion_

![image](https://github.com/user-attachments/assets/519c88d8-9b18-4e88-b797-5fcd038d66d6)

## 02 - Logistic regression

Implement logistic regression to predict whether a factory's microchip is eligible to be marketed.  
Raw data contains 3 column : col 1 and col 2 is feature and col 3 is lable.  
Traing data ( 28 degrees ): Need to mapping from raw data.

- **compute_cost**
- **compute_gradient**
- **gradient_descent**
- **predict** :

* **evaluate** : Evaluate the model's prediction results based on the following metrics : _accuracy, precision, recall and F1-score_ (similar to the classification_report method of scikit-learn)
