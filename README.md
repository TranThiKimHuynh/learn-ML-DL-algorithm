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

## 03 - Intro to Neural Network

- Create a simple deep neural network
- Tune the hyperparameters for a simple deep neural network.

### Dataset

[Calfifornia Housing Dataset](https://developers.google.com/machine-learning/crash-course/california-housing-data-description)

## 04 - Convolutional neural network

Implemented Multi class classifiaction with MNIST

- Understand the classic MNIST problem

- Create a deep neural network that performs multi-class classification

- Tune the deep neural network using CNN

### Dataset

The MNIST dataset is a large collection of handwritten digits that is commonly used for training image classification models.

- 60 000 training examples
- 10 000 testing examples  
   Each example consists of a 28x28 pixel map representing a handwritten digit and corresponding label indicating the digit's value (0-9).
  ![image](https://github.com/user-attachments/assets/79a3b427-bfbc-4950-b57b-bda873b86d8d)

### Model

This project uses a deep neural network for classification. The model consists of:

- An input layer to receive the pixel data
- Hidden layers with activation functions to learn complex patterns
- An output layer with a softmax activation function to produce probabilities for each digit class

### Optimization

The project explores various techniques to optimize the model's performance, including:

- Experimenting with the number of hidden layers and nodes
- Tuning the dropout regularization rate
- Adjusting hyperparameters like learning rate, epochs, and batch size
  ![image](https://github.com/user-attachments/assets/424e9693-dab3-4bd2-9495-8af8a72dad1b)
