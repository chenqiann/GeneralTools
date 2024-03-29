# import
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib.font_manager import FontProperties

# import external font
song = FontProperties(fname='./simsun-bold.ttf')

# raw data
x = [64, 128, 256, 512, 1024]
y1 = [0.64, 1.08, 122.29, 329.68, 697.34]
# np.random.seed(2)
# y2 = np.random.randn(10)
y2 = [7.98, 24.56, 178.76, 403.22, 1016.25]
# np.random.seed(3)
# y3 = np.random.randn(10)

# interp data
spl = make_interp_spline(x, y1, k=3)
spl2 = make_interp_spline(x, y2, k=3)
# spl3 = make_interp_spline(x, y3, k=3)
x_smooth = np.linspace(50, 1030, 10)
y1_smooth = spl(x_smooth)
y2_smooth = spl2(x_smooth)
# y3_smooth = spl3(x_smooth)


# plot
# plt.figure(figsize=(12, 10), dpi=72)
plt.figure(figsize=(12, 10), dpi=50)
# theme #
# print(plt.style.available) # 查看所有主题
# plt.style.use('seaborn-whitegrid')
# mpl.rcParams['xtick.color'] = 'red' # 可根据需要修改该主题某些属性，属性通过mpl.style.library['seaborn']查看
# mpl.rcParams['ytick.color'] = 'blue'

line1 = plt.plot(x_smooth, y1_smooth, c='green', label='Our hierarchical framework', linewidth=2)  # 这里的颜色参数cmap可以使用colormap
line2 = plt.plot(x_smooth, y2_smooth, c='blue', label='Original path-based method', linewidth=2)
# line3 = plt.plot(x_smooth, y3_smooth, c='blue', label='label3', linewidth=3)

# 网格
plt.grid(True)
# x轴名称
plt.xlabel(u'Height of images (pixels)', family='Times New Roman', fontsize=20)
# plt.xlabel(u'名称', fontsize=20, family='simsun', style='oblique', weight='bold')
# y轴名称
plt.ylabel(u'Processing time (s)', family='Times New Roman', fontsize=20)
# 图标题
plt.title('title', fontsize=24)
# 标签图例
plt.legend(loc='lower right', prop={'size': 15, "family": "Times New Roman"}, shadow=True, edgecolor='black')
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

# 坐标区间
plt.axis([50, 1200, 0, 1100])
# 坐标轴其他参数设置
plt.tick_params(axis='both', which='major', labelsize=15, left=False, bottom=False)  # 参数left，bottom等是刻度的可见设置
# 自定义坐标轴刻度值
# plt.yticks(np.arange(4), ('10$^0$', '10$^1$', '10$^2$', '10$^3$'))
# 刻度间隔
x_major_locator = plt.MultipleLocator(200) #把x轴的刻度间隔设置为10，并存在变量里
y_major_locator = plt.MultipleLocator(200) #把y轴的刻度间隔设置为1，并存在变量里
ax = plt.gca()  # ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator) #把x轴的主刻度设置为10的倍数
ax.yaxis.set_major_locator(y_major_locator) #把y轴的主刻度设置为1的倍数

# plt.show()

filepath = "figs/line_example.png"
plt.savefig(filepath)
plt.show()