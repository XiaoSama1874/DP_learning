# import matplotlib.pyplot as plt
# import numpy as np
# def run():
#     labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
#     sizes = [15, 30, 45, 10]
#     explode = (0, 0.1, 0.05, 0)
#     plt.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=270)
#     plt.show()
# def standard_method():
#     figure=plt.figure()
#     ax=figure.add_subplot(2,1,1)
#
#     ax.plot()
#     ax.contour()
#     ax.contourf()
#     ax.scatter()
#     ax.bar()
#     ax.pie()
#     ax.annotate()
#     ax.text()
#     ax.set_title()
#     ax.set_ylabel()
#     ax.set_xticks()
#     ax.set_xlim()
#
#     ax.grid()
#     ax.set_facecolor()
#     ax.set_zorder()
#     ax.set_alpha()
# if __name__ == '__main__':
#     run()

import numpy as np
import matplotlib.pyplot as plt


def boxplot():
    sample1 = np.random.normal(0, 1, 100)
    sample2 = np.random.normal(loc=1, scale=2, size=100)
    sample3 = np.random.normal(loc=0.3, scale=1.2, size=100)
    ax = plt.gca()
    ax.boxplot((sample1, sample2, sample3))
    ax.set_xticklabels(["sample1", "sample2", "sample3"])
    plt.show()


def examples():
    e1 = np.array([.3, .2, .26, .4, .36, .4])
    e2 = np.array([.4, .13, .2, .31, .33, .27])
    e3 = np.array([.25, .4, .39, .28, .26, .13])
    ax = plt.gca()
    ax.boxplot((e1, e2, e3))
    ax.set_xticklabels(["sample1", "sample2", "sample3"])
    plt.show()


def imageshow():
    x = np.random.randn((100, 100))
    plt.imshow()


def sin_3d():
    import numpy as np

    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    # 创建3d图形的两种方式

    # ax = Axes3D(fig)

    ax = fig.add_subplot(111, projection='3d')

    # X, Y value

    X = np.arange(-4, 4, 0.25)

    Y = np.arange(-4, 4, 0.25)

    X, Y = np.meshgrid(X, Y)  # x-y 平面的网格

    R = np.sqrt(X ** 2 + Y ** 2)

    # height value

    Z = np.sin(R)

    # rstride:行之间的跨度  cstride:列之间的跨度

    # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现

    # vmax和vmin  颜色的最大值和最小值

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

    # zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面

    # offset : 表示等高线图投射到指定页面的某个刻度

    ax.contourf(X, Y, Z, zdir='z', offset=-2)

    # 设置图像z轴的显示范围，x、y轴设置方式相同

    ax.set_zlim(-2, 2)

    plt.show()


if __name__ == '__main__':
    sin_3d()
