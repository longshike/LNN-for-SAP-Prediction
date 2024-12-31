# 本程序要实现基于液态神经网络的胰腺炎严重程度预测，与逻辑回归、决策树、随机森林和XGBoost算法进行对比。
# 程序从2024年10月27日开始编写，编写人：龙诗科，版权所有。

# 导入第三方库
import matplotlib.pyplot as plt  # 数据可视化库
import warnings, pandas as pd  # 告警库、数据处理库
import seaborn as sns  # 统计数据可视化

warnings.filterwarnings(action='ignore')  # 忽略告警

# 读取数据
df = pd.read_excel('test_data.xlsx')

# 设置图形大小
plt.figure(figsize=(20, 16))  # 可以根据需要调整这里的数值
# 数据的相关性分析
sns.heatmap(df.iloc[:,:].corr().round(2), cmap="YlGnBu", annot=True, annot_kws={"size": 4})  # 绘制热力图
plt.title('Heat map analysis results')  # 设置标题名称
plt.xticks(fontsize=5)  # 设置X轴标签字体大小
plt.yticks(fontsize=5)  # 设置Y轴标签字体大小

# 保存图像，设置分辨率为300DPI
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight'用于确保图像的边界紧凑

plt.show()  # 展示图片

