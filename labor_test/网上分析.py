import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("C:\\Users\\xiaobin\\PycharmProjects\\DP\\labor_test\\train.csv")
test = pd.read_csv("C:\\Users\\xiaobin\\PycharmProjects\\DP\\labor_test\\test.csv")
# 统一处理数据
full = train.append(test, ignore_index=True, sort=False)
# 数据分别
titanic = full[:891]
titanic_pre = full[891:]
# 111111       了解数据
# 首先查看数据# 显示数据特征数量
print(titanic.head())
# 打印各列特征的基本信息  包括数量 中心值 标准值  最小值 等一些统计特征# 显示数值特征
print(titanic.describe())

# titanic . info  打印基本信息  包括各个特征的数量以及缺失值个数等
# 查看数据缺失值情况
print(titanic.info())
print("缺失值情况")
print(titanic.isnull().sum(axis=0).sort_values(ascending=False))

# 分析数据和结果之前的关系(可视化)
# def plot_correlation_map(df):
#     corr=df.corr()
#     _,ax=plt.subplots()
#     cmap=sns.diverging_palette(220,10,as_cmap=True,center="light")
#     # 关系可视化
#     _=sns.heatmap(corr,cmap=cmap,square=True,cbar_kws={'shrink':.9},
#                   ax=ax,annot=True,annot_kws={'fontsize':15},fmt='.2f')
# plot_correlation_map(titanic)
# plt.show()

# 提出无用数据
titanic = titanic.drop(["Cabin", "PassengerId", 'Ticket'], axis=1)

# # 进行数据分析
# titanic[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False).plot(kind='bar')
# plt.show()
#
# titanic[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False).plot(kind='bar')
# plt.show()
#
# titanic[['Pclass','Sex','Survived']].groupby(['Pclass','Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)

# 处理性别
titanic['Sex'] = titanic['Sex'].map({'female': 1, 'male': 4}).astype(int)
print(titanic['Sex'].head())

titanic['Pclass*Sex'] = titanic.Pclass * titanic.Sex
print(titanic['Pclass*Sex'].head())
titanic['Pclass*Sex'] = pd.factorize(titanic['Pclass*Sex'])[0]
print(titanic['Pclass*Sex'].head())

# 处理名称
titanic['Title'] = titanic.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic['Title'], titanic['Sex'])
