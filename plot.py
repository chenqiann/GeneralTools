# import
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# data
x = np.arange(0,100,1)
y1 = np.arange(1,0,-0.01)
np.random.seed(2)
y2 = np.random.randn(100)
np.random.seed(3)
y3 = np.random.randn(100)

# plot
plt.figure(figsize=(8,6),dpi=72)
# theme #
# print(plt.style.available) # 查看所有主题
plt.style.use('seaborn')
# mpl.rcParams['xtick.color'] = 'red' # 可根据需要修改该主题某些属性，属性通过mpl.style.library['seaborn']查看
# mpl.rcParams['ytick.color'] = 'blue'
line1 = plt.plot(x,y1,c='red',label='label1',linewidth=1)
line2 = plt.plot(x,y2,c='green',label='label2',linewidth=2)
line3 = plt.plot(x,y3,c='blue',label='label3',linewidth=3)
plt.xlabel('xlabel',fontsize=14) #x轴名称
plt.ylabel('ylabel',fontsize=14) #y轴名称
plt.title('title',fontsize=24) #图标题
plt.legend(loc='upper left',prop={'size':15}) # 标签图例
# loc取值：
# 'best'            0
# 'upper right'     1
# 'upper left'      2
# 'lower left'      3
# 'lower right'     4
# 'right'           5
# 'center left'     6
# 'center right'    7
# 'lower center'    8
# 'upper center'    9
# 'center'          10
plt.axis([0,100,-5,5]) #坐标区间
plt.tick_params(axis='both',which='major',labelsize=10) #设置刻度的字号
x_major_locator = plt.MultipleLocator(10) #把x轴的刻度间隔设置为10，并存在变量里
y_major_locator = plt.MultipleLocator(1) #把y轴的刻度间隔设置为1，并存在变量里
ax=plt.gca() #ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator) #把x轴的主刻度设置为10的倍数
ax.yaxis.set_major_locator(y_major_locator) #把y轴的主刻度设置为1的倍数
plt.show()
# plt.savefig('filepath')