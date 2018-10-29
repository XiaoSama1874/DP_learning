# -*- coding:utf-8 -*-

# 忽略警告
import warnings

warnings.filterwarnings('ignore')

# 引入数据处理包
import numpy as np
import pandas as pd

# 引入算法包
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

# 引入帮助分析数据的包
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

# 可视化
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


def COMPARE_MODEL(train_valid_X, train_valid_y):
    def plot_learning_curve(model, train_valid_X, train_valid_y, title, ylim=[0.65, 0.90], cv=4, n_jobs=1,
                            train_sizes=np.linspace(.1, 1.0, 5)):
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Scores")
        train_sizes, train_score, test_score = learning_curve(model, train_valid_X, train_valid_y, cv=cv, n_jobs=n_jobs,
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
        plt.show()
        return train_scores_mean[4]

    def once_model(clf):
        clf.fit(train_X, train_y)
        y_pred = clf.predict(valid_X)
        return round(accuracy_score(y_pred, valid_y) * 100, 2)

    logreg = LogisticRegression()
    acc_log = plot_learning_curve(logreg, train_valid_X, train_valid_y, title="LogisticRegression")
    acc_log_once = once_model(logreg)

    xgbc = RandomForestClassifier()
    acc_xgbc = plot_learning_curve(xgbc, train_valid_X, train_valid_y, title="RandomForestClassifier")
    acc_xgbc_once = once_model(xgbc)

    svc = SVC()
    acc_svc = plot_learning_curve(svc, train_valid_X, train_valid_y, title="SVM")
    acc_svc_once = once_model(svc)

    knn = KNeighborsClassifier(n_neighbors=3)
    acc_knn = plot_learning_curve(knn, train_valid_X, train_valid_y, title="KNeighbors =3")
    acc_knn_once = once_model(knn)

    gaussian = GaussianNB()
    acc_gaussian = plot_learning_curve(gaussian, train_valid_X, train_valid_y, title="GaussianNB")
    acc_gaussian_once = once_model(gaussian)

    perceptron = Perceptron()
    acc_perceptron = plot_learning_curve(perceptron, train_valid_X, train_valid_y, title="Perceptron")
    acc_perceptron_once = once_model(perceptron)

    linear_svc = LinearSVC()
    acc_linear_svc = plot_learning_curve(linear_svc, train_valid_X, train_valid_y, title="LinearSVC")
    acc_linear_svc_once = once_model(linear_svc)

    sgd = SGDClassifier()
    acc_sgd = plot_learning_curve(sgd, train_valid_X, train_valid_y, title="SGDClassifier")
    acc_sgd_once = once_model(sgd)

    gbc = GradientBoostingClassifier()
    acc_gbc = plot_learning_curve(gbc, train_valid_X, train_valid_y, title="GradientBoostingClassifier")
    acc_gbc_once = once_model(gbc)

    models_once = pd.DataFrame({
        'Model': ['XGBC', 'Support Vector Machines', 'KNN', 'Logistic Regression',
                  'gaussian', 'Perceptron',
                  'Stochastic Gradient Decent', 'Linear SVC',
                  'GradientBoostingClassifier'],
        'Score': [acc_xgbc_once, acc_svc_once, acc_knn_once, acc_log_once,
                  acc_gaussian_once, acc_perceptron_once,
                  acc_sgd_once, acc_linear_svc_once, acc_gbc_once]})
    models_once = models_once.sort_values(by='Score', ascending=False)
    Models = [xgbc, svc, knn, logreg, gaussian, perceptron, sgd, linear_svc, gbc]
    return models_once, Models


if __name__ == '__main__':
    # configuration of graph
    mpl.style.use('ggplot')
    sns.set_style('white')
    pylab.rcParams['figure.figsize'] = 8, 6
    # read data
    train = pd.read_csv('C:\\Users\\xiaobin\\PycharmProjects\\DP\\labor_test\\train.csv')
    test = pd.read_csv('C:\\Users\\xiaobin\\PycharmProjects\\DP\\labor_test\\test.csv')
    full = train.append(test, ignore_index=True)  # 保证train和test的数据格式一样
    titanic = full[:891]
    titanic_pre = full[891:]

    del train, test
    # transfom into int type
    full['Cabin'] = full['Cabin'].isna().astype('int')
    full = full.drop(['PassengerId', 'Ticket'], axis=1)
    # transfom into int type
    full['Sex'] = full['Sex'].map({'female': 1, 'male': 4}).astype(int)

    # feature selection : construct new feature according to correlation
    full['Pclass*Sex'] = full.Pclass * full.Sex

    full['Pclass*Sex'] = pd.factorize(full['Pclass*Sex'])[0]
    # surname processing .
    full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    full['Title'] = full['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir'], 'Male_Rare')

    full['Title'] = full['Title'].replace(['Countess', 'Lady', 'Mlle', 'Mme', 'Ms'], 'Female_Rare')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Female_Rare": 5, 'Male_Rale': 6}
    full['Title'] = full['Title'].map(title_mapping)
    full['Title'] = full['Title'].fillna(0)

    full.head()
    # feature selection : construct new feature according to correlation
    full['FamilySize'] = full['Parch'] + full['SibSp'] + 1

    full.loc[full['FamilySize'] == 1, 'Family'] = 0
    full.loc[(full['FamilySize'] > 1) & (full['FamilySize'] < 5), 'Family'] = 1
    full.loc[(full['FamilySize'] >= 5), 'Family'] = 2

    # fill null value to age.
    dataset = full[['Age', 'Sex', 'Pclass']]

    guess_ages = np.zeros((2, 3))

    l = [1, 4]
    for i in range(len(l)):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == l[i]) & (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    print(guess_ages)

    for i in range(len(l)):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == l[i]) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[
                i, j]

    full['Age'] = dataset['Age'].astype(int)

    full.loc[full['Age'] <= 20, 'Age'] = 4
    full.loc[(full['Age'] > 20) & (full['Age'] <= 40), 'Age'] = 5
    full.loc[(full['Age'] > 40) & (full['Age'] <= 60), 'Age'] = 6
    full.loc[full['Age'] > 60, 'Age'] = 7

    full.head()
    # # feature selection : construct new feature according to correlation
    full['Pclass*Age'] = full.Pclass * full.Age

    # fill null value using mode
    freq_port = full.Embarked.dropna().mode()[0]
    full['Embarked'] = full['Embarked'].fillna(freq_port)

    full[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)

    full['Embarked'] = full['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    full.head()

    # using median fill null value
    # using boundray value to divide continues value into pieces.
    full['Fare'].fillna(full['Fare'].dropna().median(), inplace=True)
    # these valus come from graph visualization.
    full.loc[full['Fare'] <= 128.082, 'Fare'] = 0
    full.loc[(full['Fare'] > 128.082) & (full['Fare'] <= 256.165), 'Fare'] = 1
    full.loc[(full['Fare'] > 256.165) & (full['Fare'] <= 384.247), 'Fare'] = 2
    full.loc[full['Fare'] > 384.247, 'Fare'] = 3
    full['Fare'] = full['Fare'].astype(int)

    full = full.drop(['Name'], axis=1)

    full = full.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    # transform into int type
    full['Sex'] = full['Sex'].astype('int').astype('str')
    full = pd.get_dummies(full, prefix='Sex')
    full.head()
    # transform into int type
    full['Embarked'] = full['Embarked'].astype('str')
    full = pd.get_dummies(full, prefix='Embarked')

    full.head()

    full['Title'] = full['Title'].astype('str')

    full = pd.get_dummies(full, prefix='Title')

    # extract label
    full = full.drop(['Survived'], axis=1)

    train_valid_X = full[0:891]
    train_valid_y = titanic.Survived
    test_X = full[891:]
    train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, train_size=.78, random_state=0)

    # step 1-4   compare classifier and select best 4 classifier used to modify parameter further.
    # models_score, Models=COMPARE_MODEL(train_valid_X, train_valid_y)
    # print(models_score)

    # step  5 modify parameter

    from sklearn.svm import SVC
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import LinearSVC

    # svm degree =1 test score:0.78947  cross-validation :0.8345 degree=1
    # svm degree =2 test score:0.77     cross-validation :0.8563 (overfitting)
    # ----
    # gammas = np.linspace(0.005, 0.05, 10)
    # param_grid = [{'kernel': ['rbf'], 'gamma': gammas, "C": range(1, 100, 10)}]
    # ply = PolynomialFeatures(degree=2)
    # a=SVC()
    # clf = GridSearchCV(a, param_grid, cv=5)
    # pipeline = Pipeline([("ply", ply), ("clf", clf)])
    # pipeline.fit(train_valid_X,train_valid_y)
    # print(pipeline.score(train_valid_X,train_valid_y))
    # ----

    # linear SVC test score:0.785 train_score=0.840  degree=1
    # linear SVC test score:0.789  train_score=0.822 degree=2
    # ------
    # clf=LinearSVC()
    # # clf.fit(train_valid_X,train_valid_y)
    # # print(clf.score(train_valid_X,train_valid_y))
    # #linear svc degree=1
    # poly=PolynomialFeatures(degree=2)
    # pipeline = Pipeline([("ply", poly), ("clf", clf)])
    # pipeline.fit(train_valid_X,train_valid_y)
    # print(pipeline.score(train_valid_X,train_valid_y))
    # -----

    # randomforest  test score:0.775 cross-validation score:0.82
    # -----
    # entropy_thresholds=np.linspace(0,1,50)
    # gini_thresholds=np.linspace(0,0.5,50)
    # param_grid=[{'criterion':['entropy'],'min_impurity_decrease':entropy_thresholds},
    #             {'criterion':['gini'],'min_impurity_decrease':gini_thresholds},
    #             {'max_depth':range(2,10)},
    #             {'min_samples_split':range(2,30,2)}]
    # clf=BaggingClassifier()
    # clf=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
    # clf.fit(train_valid_X,train_valid_y)
    # print(clf.best_score_)
    # -----

    # logisticregression  test_score = 0.78 cv_score:0.823
    # try to add poly.degree=2     test_score=0.7848  cv_score:0.833
    # try to add poly degree=3      test_score=0.79425 cv_score:0.8496-------------the best model.
    param_grid = [{"penalty": ["l1", "l2"], "C": np.linspace(0.1, 10, 10)}]
    model = LogisticRegression()
    clf = GridSearchCV(model, param_grid)
    poly = PolynomialFeatures(degree=3)
    pipeline = Pipeline([("ply", poly), ("clf", clf)])
    pipeline.fit(train_valid_X, train_valid_y)
    print(pipeline.score(train_valid_X, train_valid_y))
    # clf.fit(train_valid_X,train_valid_y)
    # print(clf.best_score_)
    # print(clf.best_params_)

    # # score=0.779
    # poly=PolynomialFeatures(degree=3)
    #
    # bag=BaggingClassifier(LinearSVC(),n_estimators=20)
    #
    # pipeline=Pipeline([("Polynomial_Features",poly),("bag",bag)])
    #
    # pipeline.fit(train_valid_X,train_valid_y)

    # 得分 score = 0.79425
    # poly = PolynomialFeatures(degree=2)
    #
    # bag = BaggingClassifier(LogisticRegression(penalty='l1'), n_estimators=10)
    #
    # pipeline = Pipeline([("Polynomial_Features", poly), ("bag", bag)])
    #
    # pipeline.fit(train_valid_X, train_valid_y)

    # test score 0.765
    # param_grid = {'n_estimators': [50, 80, 100], 'learning_rate': [0.05, 0.08, 0.1, 0.12], 'max_depth': [3, 4]}
    # grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
    # grid_search.fit(train_valid_X, train_valid_y)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)

    passenger_id = titanic_pre.PassengerId
    test_y_svm = pipeline.predict(test_X)
    test = pd.DataFrame({'PassengerId': passenger_id, 'Survived': np.round(test_y_svm).astype('int32')})
    #
    print(test.head())
    #
    test.to_csv('logistic_regression degree=3.csv', index=False)
