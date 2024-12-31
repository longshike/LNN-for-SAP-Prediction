# 导入第三方库
import matplotlib.pyplot as plt  # 数据可视化库
import warnings, pandas as pd  # 告警库、数据处理库
from sklearn.model_selection import train_test_split  # 数据拆分工具
from sklearn.metrics import roc_curve, auc  # 模型评估方法
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类模型
from imblearn.over_sampling import SMOTE  # 导入过采样工具

warnings.filterwarnings(action='ignore')  # 忽略告警

# 读取数据
df = pd.read_excel('test_data.xlsx')

# 根据提供的顺序逐一叠加选择特征变量
order = [
    "HBDH", "LDH", "CRP", "NEU#", "WBC", "INR", "LYM%", "NEU%",
    "CREA", "CA", "UREA", "FIB", "AMY", "EOS%", "CK-MB",
    "EOS#", "HGB", "MON#", "HDL-C", "AST/ALT", "ALB", "A/G",
    "HCT", "BAS%", "PCT", "TG", "PT%", "APTT", "RDW-SD",
    "GLOB", "RBC", "CHE", "LYM#", "LDL-C", "RDW-CV", "NA", "MON%",
    "MCH", "TBIL", "PT", "PA", "DBIL", "AST", "TP",
    "CO2-L", "TBA"
]

# 初始化一个空的DataFrame，用于存储逐步选择的特征变量
X_selected = pd.DataFrame()

# 存储AUC值的列表
auc_values = []

# 逐一叠加选择特征变量
for i, column in enumerate(order):
    if column in df.columns:
        # 添加当前特征变量
        X_selected = pd.concat([X_selected, df[[column]]], axis=1)

        # 提取特征变量和标签变量
        y = df['Label']

        # 数据均衡化
        model_SMOTE = SMOTE(random_state=0)  # 建立过采样模型
        x_SMOTE_resampled, y_SMOTE_resampled = model_SMOTE.fit_resample(X_selected, y)  # 过采样后样本结果
        x_SMOTE_resampled = pd.DataFrame(x_SMOTE_resampled, columns=X_selected.columns.values)  # 构建DataFrame框架
        y_SMOTE_resampled = pd.DataFrame(y_SMOTE_resampled, columns=['Label'])  # 构建DataFrame框架
        SMOTE_resampled = pd.concat([x_SMOTE_resampled, y_SMOTE_resampled], axis=1)  # 框架合并

        # 利用过采样后的样本来进行训练
        X1 = SMOTE_resampled.drop(columns=['Label']).values  # 构建特征数据集
        y1 = SMOTE_resampled['Label'].values  # 构建标签集
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3,
                                                            random_state=42)  # 划分训练集和测试集，30%作为测试集
        # 手动设置随机森林参数
        rfc = RandomForestClassifier(
            n_estimators=100,  # 树的数量
            max_depth=10,  # 树的最大深度
            min_samples_split=2,  # 分割内部节点所需的最小样本数
            min_samples_leaf=4,  # 叶子节点所需的最小样本数
            max_features='auto',  # 最大特征数
            random_state=42  # 随机数生成器的种子
        )
        rfc.fit(X_train, y_train)  # 拟合

        # 模型预测
        probs = rfc.predict_proba(X_test)  # 返回分类的概率
        preds = probs[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, preds)  # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
        roc_auc = auc(fpr, tpr)  # 计算AUC值
        auc_values.append(roc_auc)  # 保存AUC值

    else:
        print(f"Column {column} not found in the DataFrame.")

# 绘制特征变量个数与AUC值之间的关系图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(auc_values) + 1), auc_values, marker='^', linestyle='-', color='m')
plt.title('RF:Relationship between Number of Features and AUC')
plt.xlabel('Number of Features')
plt.ylabel('AUC Value')
plt.grid(True)

# 保存图像，设置分辨率为300DPI
plt.savefig('RF_features.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight'用于确保图像的边界紧凑
plt.show()