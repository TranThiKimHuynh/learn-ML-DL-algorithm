import numpy as np 
import json
import pandas as pd
import seaborn as sns


file_path = 'training_data.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# List to store the values of X1, X2, and Y
X1_values = []
X2_values = []
Y_values = []


for line in lines:
    values = line.strip().split(',')
    X1_values.append(float(values[0]))
    X2_values.append(float(values[1]))
    Y_values.append(float(values[2]))



def map_feature(x1, x2):
#   x1, x2 type: numpy array
#   Returns a new feature array with more features, comprising of 
#   x1, x2, x1.^2, x2.^2, x1*x2, x1*x2.^2, etc.

    degree = 6
    out = np.ones([len(x1), int((degree + 1) * (degree + 2) / 2)])
    idx = 1

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            a1 = x1 ** (i - j)
            a2 = x2 ** j
            out[:, idx] = a1 * a2
            idx += 1

    return out

class LogisticRegression:
    def __init__(self, Alpha=0.5, Lambda =1 ,Iters=10, verbose=False):
        self.Alpha = Alpha
        self.Lambda = Lambda
        self.Iters = Iters
        self.verbose = verbose
        self.theta = None
    
    def compute_cost(self, X, y, theta):
        m = len(y)
        h_theta = 1 / (1 + np.exp(-np.dot(X, theta)))
        J = (np.sum(-y * np.log(h_theta) - (1 - y) * np.log(1 - h_theta))) / m + self.Lambda * np.sum(theta[1:] ** 2) / (2 * m)
        return J
    
    def compute_gradient(self, X, y, theta,j):
        m = len(y)
        h_theta = 1 / (1 + np.exp(-np.dot(X, theta)))
        loss = h_theta - y
        dJ = np.dot(X.T, loss) / m
        dJ[1:] += (self.Lambda / m) * theta[1:]
        J = self.compute_cost(X, y, theta)
        return (J, dJ)
    
    def gradient_descent(self, X, y):
        X = np.c_[np.ones(len(X),dtype='int64'), X]
        theta = np.zeros(X.shape[1])
        print(f'The total of training sample: {len(y)}')
        for i in range(self.Iters):
            J, dJ = self.compute_gradient(X,y,theta,i)

            theta = theta - self.Alpha*dJ
            if self.verbose:
                print(f'Iter {i + 1}, loss = {self. compute_cost(X, y, theta)}')
        return theta
    
    def fit(self, X, y):
        self.theta = self.gradient_descent(X, y)
    
    def predict(self, X):
        results = []
        for Xi in X:
            prediction = 1/ (1 + np.exp(-np.dot(np.insert(Xi, 0, 1), self.theta)))
            if prediction >= 0.5 :
                prediction = 1
            else:
                prediction = 0
            results.append(prediction)
            
        return results

    def evaluate(self, X, y):
        y_predict = self.predict(X)
        accuracy = np.mean(y_predict == y)
        n = len(y)

        true_positive = 0
        predicted_positive = 0
        positive = 0
        for i in range(n):
            if y[i] == 1 and y_predict[i] == 1:
                true_positive += 1
            if y_predict[i] == 1:
                predicted_positive += 1
            if y[i] == 1:
                positive += 1

        precision = true_positive / predicted_positive
        recall = true_positive / positive
        f1_score = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1_score
    

#Load config
with open('config.json',) as f:
    configs = json.load(f)

# Convert the lists to numpy arrays
X1 = np.array(X1_values)
X2 = np.array(X2_values)
Y = np.array(Y_values)
X = map_feature(X1, X2)


model = LogisticRegression(Alpha= configs['Alpha'],Lambda=configs['Lambda'], Iters=configs['NumIter'], verbose=False)
model.fit(X, Y)


# Use the predict method to predict for the sample
y_predict = model.predict(X)


# Evaluate the model
classificationn_report = model.evaluate(X, Y)


# Save the model to a file model.json
model_dict = {
    'theta': model.theta.tolist()
}
with open('model.json', 'w') as f:
    json.dump(model_dict, f)

# Save predict and evaluate of training date to classification_report.json
classification_report = {
    'y_predict': y_predict,
    'classification_report': {
        'accuracy': classificationn_report[0],
        'precision': classificationn_report[1],
        'recall': classificationn_report[2],
        'f1_score': classificationn_report[3]
    }
}
with open('classification_report.json', 'w') as f:
    json.dump(classification_report, f,indent=4)




