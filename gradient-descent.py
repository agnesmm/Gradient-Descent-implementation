import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

# load and prepare the data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=['prices'])

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42)

def prepare_dataset(X, y):
  X = pd.DataFrame(scaler.transform(X))
  X.insert(0, 'x0', np.ones(X.shape[0]))
  y = y.prices.values
  return X, y

# prepare data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train, y_train = prepare_dataset(X_train, y_train)

# initialize
n = X.shape[1] + 1
m = X_train.shape[0]
theta = np.ones(n)
alpha = 0.1
iterations_number = 100

# main functions
def hypothesis(X, theta):
  return np.dot(X, theta)

def cost_function(y, hypothesis):
  return np.sum((hypothesis - y)**2)/(2*y.shape[0])

#process to gradient descent algorithm
def gradient_descent(theta, plot=False):
  cost = []
  for a in range(iterations_number):
    if plot:
      cost.append(cost_function(y_train, hypothesis(X_train, theta)))
    if a%10==0:
      print cost_function(y_train, hypothesis(X_train, theta))
    h = hypothesis(X_train, theta)
    for j in range (n):
      Xj = X_train.iloc[:,j]
      theta[j] = theta[j] - (alpha/m)*np.multiply(h-y_train, Xj).sum()
  if plot:
    plt.plot(cost)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost function')
    plt.show()
  return theta

theta = gradient_descent(theta, plot=True)
X_test, y_test = prepare_dataset(X_test, y_test)
print 'cost function on test set:', cost_function(y_test, hypothesis(X_test, theta))








