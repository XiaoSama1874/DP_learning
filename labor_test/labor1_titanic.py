import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def get_data():
    df = pd.read_csv("C:\\Users\\xiaobin\\PycharmProjects\\DP\\scikit-learn\\descision_tree\\train.csv", index_col=0)
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df['Sex'] = (df['Sex'] == 'male').astype('int')
    labels = df['Embarked'].unique().tolist()
    df['Embarked'] = df['Embarked'].apply(lambda n: labels.index(n))
    df = df.fillna(0)
    return df


train = get_data()
y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.2)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
# entropy_thresholds=np.linspace(0,1,50)
# gini_thresholds=np.linspace(0,0.5,50)
#
# param_grid=[{'criterion':['entropy'],'min_impurity_decrease':entropy_thresholds},
#             {'criterion':['gini'],'min_impurity_decrease':gini_thresholds},
#             {'max_depth':range(2,10)},
#             {'min_samples_split':range(2,30,2)}]
# clf=BaggingClassifier()
# clf=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
#
# clf.fit(X_train,Y_train)
# print("{}   {}".format("RandomTreeClassifier",clf.best_params_))
# print(clf.best_score_)
#
#
#
# # clf=RandomForestClassifier(**clf.best_params_)
# # clf.fit(X_train,Y_train)
# test_score=clf.score(X_test,Y_test)
# # print("Train score {}  Test score {} ".format(train_score,test_score))
# print("Test score {} ".format(test_score))
#
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

gammas = np.linspace(0.005, 0.05, 10)

param_grid = [{'kernel': ['rbf'], 'gamma': gammas, "C": range(1, 100, 10)}]
ply = PolynomialFeatures(degree=2)
clf = GridSearchCV(SVC(), param_grid, cv=5)
pipeline = Pipeline([("ply", ply), ("clf", clf)])
clf.fit(X, y)
print("{}   {}".format(clf.best_estimator_, clf.best_params_))
print(clf.best_score_)
print(clf.score(X_test, Y_test))

#
# param_grid=[{'kernel':['poly'],'degree':range(1,2)}]
# clf=GridSearchCV(SVC(),param_grid,cv=5)
# clf.fit(X_train,Y_train)
# print(clf.best_params_)
# print(clf.best_score_)
# print(clf.score(X_test,Y_test))
#

# 使用逻辑回归来做
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures
#
# param_grid=[{"penalty":['l1']},{"penalty":['l2']}]
# clf=GridSearchCV(LogisticRegression(),param_grid,cv=5)
# clf.fit(X_train,Y_train)
# test_score=clf.score(X_test,Y_test)
# print(clf.best_params_)
# print(clf.best_score_)
# print(test_score)


# 尝试使用多项式增加特征
# poly=PolynomialFeatures(degree=2)
# pipeline=Pipeline([("Polynomial_Features",poly),("LogisticRegression",LogisticRegression(penalty='l1'))])
# pipeline.fit(X_train,Y_train)
# print("degree=2")
# print(pipeline.score(X_train,Y_train))
# print(pipeline.score(X_test,Y_test))
#
#
# poly=PolynomialFeatures(degree=3)
# pipeline=Pipeline([("Polynomial_Features",poly),("LogisticRegression",LogisticRegression(penalty='l1'))])
# pipeline.fit(X_train,Y_train)
# print("degree=3")
# print(pipeline.score(X_train,Y_train))
# print(pipeline.score(X_test,Y_test))
# logistic_regression=pipeline.named_steps['LogisticRegression']
# print(logistic_regression.coef_.shape)
# print(np.count_nonzero(logistic_regression.coef_))
#


# 选择使用自助聚合算法

# from  sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import BaggingClassifier
# # 尝试使用多项式增加特征
# print("bagging Classifier degree=2")
# poly=PolynomialFeatures(degree=2)
# bag=BaggingClassifier(LogisticRegression(penalty='l1'),n_estimators=20)
# pipeline=Pipeline([("Polynomial_Features",poly),("bag",bag)])
# pipeline.fit(X_train,Y_train)
# print(pipeline.score(X_train,Y_train))
# print(pipeline.score(X_test,Y_test))
#
# print("bagging Classifier degree=3")
# poly=PolynomialFeatures(degree=3)
# bag=BaggingClassifier(LogisticRegression(penalty='l1'),n_estimators=20)
# pipeline=Pipeline([("Polynomial_Features",poly),("bag",bag)])
#
# pipeline.fit(X_train,Y_train)
# print(pipeline.score(X_train,Y_train))
# print(pipeline.score(X_test,Y_test))


# from sklearn.naive_bayes import  MultinomialNB
# from sklearn.naive_bayes import  GaussianNB
# print("MultinomialNB ")
# nv=MultinomialNB()
# nv.fit(X_train,Y_train)
# a1=nv.score(X_train,Y_train)
# a2=nv.score(X_test,Y_test)
# print("train score {} test score {}".format(a1,a2))
#
# print("GassianNB")
# nv=GaussianNB()
# nv.fit(X_train,Y_train)
# a1=nv.score(X_train,Y_train)
# a2=nv.score(X_test,Y_test)
# print("train score {} test score {}".format(a1,a2))
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
