import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 定义数据
algorithms = ['LR', 'DCT', 'RF', 'XGBoost', 'LNN']
auc_without_smote = [0.877037562012757, 0.7563193952279708, 0.8577840774864162, 0.8420741790692181, 0.8366406803685329]
auc_with_smote = [0.891033, 0.868372, 0.92245, 0.907489, 0.965854]

# 指定颜色
color_without_smote = '#5CACEE'  # 未采用SMOTE技术的颜色（森林绿）
color_with_smote = '#FF7F50'    # 采用SMOTE技术的颜色（金盏花黄）

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
bars1 = ax.bar(bar1, auc_without_smote, bar_width, label='No SMOTE technology', color=color_without_smote)
bars2 = ax.bar(bar2, auc_with_smote, bar_width, label='With SMOTE technology', color=color_with_smote)

# 设置图例，横排显示
ax.legend(loc='upper center', ncol=2)  # 设置图例位置和列数

# 添加标签和标题
ax.set_xlabel('Different algorithms')
ax.set_ylabel('AUC value')
ax.set_title('AUC comparison of different algorithms (using SMOTE or not)')
ax.set_xticks(index)
ax.set_xticklabels(algorithms, rotation=45)  # 将x轴标签旋转45度，便于阅读

# 美化图表
ax.grid(False)
plt.tight_layout()  # 调整整体空白

# 保存图像，设置分辨率为300DPI
plt.savefig('SMOTE_bar.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight'用于确保图像的边界紧凑
plt.show()