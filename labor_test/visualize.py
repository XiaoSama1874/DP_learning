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


def plot_correlation_map(df):
    corr = df.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 15},
        fmt='.2f'
    )


# 配置可视化
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 8, 6

train = pd.read_csv('C:\\Users\\xiaobin\\PycharmProjects\\DP\\labor_test\\train.csv')
test = pd.read_csv('C:\\Users\\xiaobin\\PycharmProjects\\DP\\labor_test\\test.csv')
full = train.append(test, ignore_index=True)  # 保证train和test的数据格式一样
titanic = full[:891]
titanic_pre = full[891:]
del train, test
# plot_correlation_map(titanic)
plt.show()
# visualize Pclass with Survival
# titanic[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False).plot(kind='bar')
titanic[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False).plot(
    kind='bar')
# labels = titanic['Embarked'].unique().tolist()
# titanic['Embarked'] = titanic['Embarked'].apply(lambda n: labels.index(n))
# titanic[["Embarked", "Survived"]].groupby(['Survived'], as_index=False).mean().sort_values(by='Survived', ascending=False).plot(kind='bar')
plt.show()
freq_port = titanic.Embarked.dropna().mode()[0]
titanic['Embarked'] = titanic['Embarked'].fillna(freq_port)

titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                           ascending=False)

titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
titanic.head()

titanic['Pclass*Sex'] = titanic.Pclass * titanic.Sex
titanic['Pclass*Sex'] = pd.factorize(titanic['Pclass*Sex'])[0]
