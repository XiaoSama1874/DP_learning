# Package imports
import numpy as np
import matplotlib.pyplot as plt
from Andrew.week_2_one_hidden_layers.testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Andrew.week_2_one_hidden_layers.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, \
    load_extra_datasets

np.random.seed(1)  # set a seed so that the results are consistent

X, Y = load_planar_dataset()

### START CODE HERE ### (≈ 3 lines of code)

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]  # training set size

### END CODE HERE ###


# Train the logistic regression classifier

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel())

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      '% ' + "(percentage of correctly labelled datapoints)")

plt.show()
