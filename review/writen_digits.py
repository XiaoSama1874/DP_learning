from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# 查看数据结构  俩种方式
digits = load_digits()
# print(digits.data.shape)
# print(digits.target[:10])
# print(digits.images[0])
# print(digits.target_names)


X, y = load_digits(return_X_y=True)


def plot_digits(sample, c):
    plt.title("Digit " + c)
    plt.axes('off')
    plt.imshow(sample, cmap=plt.cm.gray)


# 俩行四列显示八个数据
for index, image in enumerate(X[:8]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image.reshape(8, -1), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Digit : {}".format(y[index]), fontsize=20)
plt.show()

# 开始选择模型 选择 SVM
# param_grid=[{"kernel":("poly","rbf"),"gamma":np.linspace(.0001,.01,10),"degree":range(2,3)}]
# new_clf=GridSearchCV(SVC(C=1.0,kernel='rbf'),param_grid,cv=5)
# new_clf.fit(X,y)
# print(new_clf.best_score_)
# print(new_clf.best_params_)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0, penalty='l1')
print(cross_val_score(clf, X, y, cv=5))
a = cross_val_score()

fig, axes = plt.subplots()
plt.subplots_adjust(hspace=.1, wspace=.1)
for i, ax in enumerate(axes.flat):
    ax.imshow()
    ax.text()
    ax.set_xticks([])
    ax.set_xticks([])
