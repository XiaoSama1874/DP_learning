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

if __name__ == '__main__':
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
    print('DataSets: ', 'full: ', full.shape, 'titanic: ', titanic.shape)
