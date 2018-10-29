# import numpy as np
# import matplotlib.pyplot as plt
# x=[1,2,3,4,5,6,7,8,9]
# y=[0.199,0.389,0.580,0.783,0.980,1.177,1.380,1.575,1.771]
# A=np.vstack([x,np.ones(len(x))]).T
#
# # 调用最小二乘法
# a,b=np.linalg.lstsq(A,y)[0]
#
# # 转化为numpy array
# x=np.array(x)
# y=np.array(y)
# # 画图
# plt.plot(x,y,'o',label='Original data',markersize=10)
# plt.plot(x,a*x+b,'r',label='Fitted line')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2 + 1

# 这条语句会默认创建
plt.figure()

# 使用 plot 画线条  支持一次性画多条线    ‘b-’ b 代表color - 代表 linestyle
# # plt.plot(x,y1,'r-',x,y2,'b-')
# l1,=plt.plot(x,y2)
# l2,=plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
plt.plot(x, y2, label='up')
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='down')
# 设置x轴  y轴的范围
# plt.xlim((-1,2))
# plt.ylim((-2,3))
# 设置x y的名字
plt.xlabel('I am x')
plt.ylabel('I am y')

# 设置 标尺
new_ticks = np.linspace(-1, 2, 5)
# plt.xticks(new_ticks)
# 设置带名字的标尺
# r 表示正则 反斜杠用来转移 $$ 符号用来表示字体
plt.yticks([-2, -1.8, -1, 1.22, 3],
           [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])

# gac='get current axis'
# 得到当前的axis
ax = plt.gca()

# spines 意味 图形四周的'轴'  脊椎

# 设置颜色 为 none  消失  移去 上 右 俩根 '轴'

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 设置与图像对应的 x  轴  y 轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# 设置位置 data  表示 采用ｘ，ｙ　轴的值  可以使用 axis　　使用百分比表示

ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# 设置名牌
# 俩种方式设定
# 第一种 plt.legend(handles=[],labels=[],loc=)
# 第二种 需要在画线时添加 lable 指定属性 直接使用 plt.legend(loc='')

plt.legend(loc='best')
# plt.legend(handles=[l1,l2],labels=['up','down'],loc='lower right')


# 添加文字注释 text,annotation
x0 = 1
y0 = 2 * x0 + 1
plt.plot([x0, x0, ], [0, y0], 'k--', lw=2.5)

# scatter 用来画散点图 s 表示尺寸 color 表示 颜色
plt.scatter([x0, ], [y0, ], s=50, color='b')

# 针对某个点添加注释
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

# 直接添加文本
plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})

# 设置 标尺刻度的可见性
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    # set zorder for ordering the plot in plt 2.0.2 or higher
    label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, zorder=2))

plt.show()

# 创建 3d 图像
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# fig=plt.figure()
# ax=Axes3D(fig)
#
# #
# # x , y value
# x=np.arange(-4,4,0.25)
# y=np.arange(-4,4,0.25)
#
# X,Y=np.meshgrid(x,y)
# R=np.sqrt(X**2+Y**2)
# # height value
# Z=np.sin(R)
#
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
#
# # plt.get_cmap('rainbow')
# # 投影
# ax.contourf(X,Y,Z,zdir='z',offset=-1,cmap=plt.get_cmap('rainbow'))
# ax.contourf(X,Y,Z,zdir='y',offset=4,cmap=plt.get_cmap('rainbow'))
#
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.zaxis.set_ticks_position('right')
#
# ax.spines['bottom'].set_position(('data',0))
# ax.spines['left'].set_position(('data',0))
# ax.spines['right'].set_position(('data',0))
#
# plt.show()
