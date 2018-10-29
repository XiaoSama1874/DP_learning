from sklearn.datasets import load_boston
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, Y_tarin, Y_test = train_test_split(X, y, test_size=.2, random_state=2)


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("Polynomial_Features", polynomial_features), ("Linear_regression", linear_regression)])
    return pipeline


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Scores")
    train_sizes, train_score, test_score = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                          train_sizes=train_sizes)

    print(train_score)
    train_scores_mean = np.mean(train_score, axis=1)
    train_score_std = np.std(train_score, axis=1)
    test_scores_mean = np.mean(test_score, axis=1)
    test_score_std = np.std(test_score, axis=1)
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score ")
    plt.legend(loc="best")
    return plt


#
# model=polynomial_model(2)
# start=time.clock()
# model.fit(X_train,Y_tarin)
# train_score=model.score(X_train,Y_tarin)
# cv_score=model.score(X_test,Y_test)
# print('elapse :{} ; train_score :{} ; test_score:{}'.format(time.clock()-start,train_score,cv_score))

cv = ShuffleSplit(n_splits=10, test_size=.2, random_state=0)
plt.figure(figsize=(18, 4))
title = 'Learning Curves (degree={})'
degrees = [1, 2, 3]
start = time.clock()
for i in range(len(degrees)):
    plt.subplot(1, 3, i + 1)
    plot_learning_curve((polynomial_model(degrees[i])), X, y, title.format(degrees[i]), ylim=(.01, 1.01), cv=cv)
plt.show()
