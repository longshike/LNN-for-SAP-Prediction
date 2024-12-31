import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 定义数据
algorithms = ['LR', 'DCT', 'RF', 'XGBoost', 'LNN']
auc_without_feature_selection = [0.887621832358675, 0.83180636777128, 0.910038986354776, 0.884877, 0.965854]
auc_with_feature_selection = [0.891033, 0.868372, 0.92245, 0.907488628979857, 0.9684860298895386]

# 指定颜色
color_without_feature_selection = '#1f77b4'  # 未经过特征选取的颜色
color_with_feature_selection = '#ff7f0e'    # 经过特征选取的颜色

# 设置柱状图的宽度和间隔
bar_width = 0.2
spacing = 0.1

# 计算柱状图的位置
n_groups = len(algorithms)
index = np.arange(n_groups)
bar1 = index - bar_width/2
bar2 = index + bar_width/2

# 创建柱状图
fig, ax = plt.subplots()
bars1 = ax.bar(bar1, auc_without_feature_selection, bar_width, label='Not Feature Selected', color=color_without_feature_selection, hatch='')
bars2 = ax.bar(bar2, auc_with_feature_selection, bar_width, label='After feature selection', color=color_with_feature_selection, hatch='')

# 设置图例，横排显示
ax.legend(loc='upper center', ncol=2)  # 设置图例位置和列数

# 添加标签和标题
ax.set_xlabel('Different algorithms')
ax.set_ylabel('AUC value')
ax.set_title('AUC comparison of different modeling algorithms (feature selection or not)')
ax.set_xticks(index)
ax.set_xticklabels(algorithms, rotation=45)  # 将x轴标签旋转45度，便于阅读

# 美化图表
#ax.grid(True)
plt.tight_layout()  # 调整整体空白

# 保存图像，设置分辨率为300DPI
plt.savefig('Features_bar.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight'用于确保图像的边界紧凑
plt.show()