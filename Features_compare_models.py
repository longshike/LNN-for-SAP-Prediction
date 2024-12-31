# 本程序要实现基于液态神经网络的胰腺炎严重程度预测，与逻辑回归、决策树、随机森林和XGBoost算法进行对比。
# 程序从2024年10月27日开始编写，编写人：龙诗科，版权所有。

# 导入第三方库
import matplotlib.pyplot as plt  # 数据可视化库
import warnings, pandas as pd  # 告警库、数据处理库
from sklearn.model_selection import train_test_split  # 数据拆分工具
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc # 模型评估方法
from xgboost import XGBClassifier  # 导入XGBoost分类器
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类模型
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
from sklearn import metrics  # 导入模型评估指标工具
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类模型
from imblearn.over_sampling import RandomOverSampler, SMOTE  #导入过采样工具
import random


warnings.filterwarnings(action='ignore')  # 忽略告警

# 读取数据
df = pd.read_excel('test_data.xlsx')

# 提取特征变量和标签变量（不同方法提取不同的特征变量）
#原始的数据，所有变量都提取
y = df.Label
X = df.drop('Label', axis=1)

#logic回归提取特征变量
# 指定要读取的特征变量列名
lr_feature_columns = [
    "HBDH", "LDH", "CRP", "NEU#", "INR", "LYM%", "NEU%", "CA", "UREA", "AMY", "EOS%",
    "EOS#", "HGB", "ALB", "BAS%", "PCT", "TG", "APTT", "RDW-SD",
    "GLOB", "CHE", "LYM#", "NA", "MON%",
    "MCH", "TBIL", "TP","CO2-L"
]
y_lr = df.Label
X_lr = df[lr_feature_columns]

#决策树回归提取特征变量
# 指定要读取的特征变量列名
dct_feature_columns = [
    "HBDH", "LDH", "NEU#", "WBC", "CA", "FIB", "AMY", "CK-MB",
    "MON#", "HDL-C", "A/G","HCT", "BAS%", "TG", "RDW-SD",
    "CHE", "LDL-C", "MCH", "DBIL", "AST", "TP", "CO2-L"
]
y_dct = df.Label
X_dct = df[dct_feature_columns]

#随机森林回归提取特征变量
# 指定要读取的特征变量列名
rfc_feature_columns = [
    "HBDH", "LDH", "CRP", "NEU#", "WBC", "INR", "NEU%",
    "CREA", "CA", "AMY", "EOS%", "CK-MB",
    "HGB", "HDL-C", "A/G",
    "BAS%", "TG", "APTT", "RDW-SD",
    "GLOB", "CHE", "RDW-CV", "MON%",
    "TBIL", "PA", "DBIL", "TP",
    "CO2-L", "TBA"
]
y_rfc = df.Label
X_rfc = df[rfc_feature_columns]

#XGBoost回归提取特征变量
# 指定要读取的特征变量列名
xgbc_feature_columns = [
    "HBDH", "LDH", "CRP", "NEU#", "WBC", "INR", "NEU%",
    "CREA", "CA", "UREA", "AMY", "EOS%", "CK-MB",
    "EOS#", "HGB", "HDL-C", "ALB", "A/G",
    "BAS%", "PCT", "PT%", "APTT", "RDW-SD",
    "GLOB", "CHE", "LYM#", "RDW-CV",
    "MCH", "TBIL", "PT", "DBIL", "AST", "TP",
    "CO2-L"
]
y_xgbc = df.Label
X_xgbc = df[xgbc_feature_columns]
#X_xgbc = df.drop('Label', axis=1)

#液态神经网络回归提取特征变量
# 指定要读取的特征变量列名
lnn_feature_columns = [
    "HBDH", "CRP", "NEU#", "WBC", "INR",
    "CREA", "CA", "UREA", "FIB", "AMY", "EOS%",
    "HGB", "HDL-C", "A/G",
    "BAS%", "TG", "APTT", "RDW-SD",
    "RBC", "CHE", "LDL-C", "RDW-CV", "MON%",
    "MCH", "PT", "DBIL",
    "CO2-L"
]
y_lnn = df.Label
X_lnn = df[lnn_feature_columns]

#数据均衡化，整体的
model_SMOTE = SMOTE(random_state=0)  # 建立过采样模型
x_SMOTE_resampled, y_SMOTE_resampled = model_SMOTE.fit_resample(X, y)  # 过采样后样本结果
x_SMOTE_resampled = pd.DataFrame(x_SMOTE_resampled, columns=X.columns.values)  # 构建DataFrame框架
y_SMOTE_resampled = pd.DataFrame(y_SMOTE_resampled, columns=['Label'])  # 构建DataFrame框架
SMOTE_resampled = pd.concat([x_SMOTE_resampled, y_SMOTE_resampled], axis=1)  # 框架合并
#logic回归
x_SMOTE_resampled_lr, y_SMOTE_resampled_lr = model_SMOTE.fit_resample(X_lr, y_lr)  # 过采样后样本结果
x_SMOTE_resampled_lr = pd.DataFrame(x_SMOTE_resampled_lr, columns=X_lr.columns.values)  # 构建DataFrame框架
y_SMOTE_resampled_lr = pd.DataFrame(y_SMOTE_resampled_lr, columns=['Label'])  # 构建DataFrame框架
SMOTE_resampled_lr = pd.concat([x_SMOTE_resampled_lr, y_SMOTE_resampled_lr], axis=1)  # 框架合并
#决策树
x_SMOTE_resampled_dct, y_SMOTE_resampled_dct = model_SMOTE.fit_resample(X_dct, y_dct)  # 过采样后样本结果
x_SMOTE_resampled_dct = pd.DataFrame(x_SMOTE_resampled_dct, columns=X_dct.columns.values)  # 构建DataFrame框架
y_SMOTE_resampled_dct = pd.DataFrame(y_SMOTE_resampled_dct, columns=['Label'])  # 构建DataFrame框架
SMOTE_resampled_dct = pd.concat([x_SMOTE_resampled_dct, y_SMOTE_resampled_dct], axis=1)  # 框架合并
#随机森林
x_SMOTE_resampled_rfc, y_SMOTE_resampled_rfc = model_SMOTE.fit_resample(X_rfc, y_rfc)  # 过采样后样本结果
x_SMOTE_resampled_rfc = pd.DataFrame(x_SMOTE_resampled_rfc, columns=X_rfc.columns.values)  # 构建DataFrame框架
y_SMOTE_resampled_rfc = pd.DataFrame(y_SMOTE_resampled_rfc, columns=['Label'])  # 构建DataFrame框架
SMOTE_resampled_rfc = pd.concat([x_SMOTE_resampled_rfc, y_SMOTE_resampled_rfc], axis=1)  # 框架合并
#XGBoost
x_SMOTE_resampled_xgbc, y_SMOTE_resampled_xgbc = model_SMOTE.fit_resample(X_xgbc, y_xgbc)  # 过采样后样本结果
x_SMOTE_resampled_xgbc = pd.DataFrame(x_SMOTE_resampled_xgbc, columns=X_xgbc.columns.values)  # 构建DataFrame框架
y_SMOTE_resampled_xgbc = pd.DataFrame(y_SMOTE_resampled_xgbc, columns=['Label'])  # 构建DataFrame框架
SMOTE_resampled_xgbc = pd.concat([x_SMOTE_resampled_xgbc, y_SMOTE_resampled_xgbc], axis=1)  # 框架合并
#LNN
x_SMOTE_resampled_lnn, y_SMOTE_resampled_lnn = model_SMOTE.fit_resample(X_lnn, y_lnn)  # 过采样后样本结果
x_SMOTE_resampled_lnn = pd.DataFrame(x_SMOTE_resampled_lnn, columns=X_lnn.columns.values)  # 构建DataFrame框架
y_SMOTE_resampled_lnn = pd.DataFrame(y_SMOTE_resampled_lnn, columns=['Label'])  # 构建DataFrame框架
SMOTE_resampled_lnn = pd.concat([x_SMOTE_resampled_lnn, y_SMOTE_resampled_lnn], axis=1)  # 框架合并

#利用过采样后的样本来进行训练
#整体的
X1 = SMOTE_resampled.drop(columns=['Label']).values  # 构建特征数据集
y1 = SMOTE_resampled['Label'].values  # 构建标签集
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
#logic回归
X1_lr = SMOTE_resampled_lr.drop(columns=['Label']).values  # 构建特征数据集
y1_lr = SMOTE_resampled_lr['Label'].values  # 构建标签集
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X1_lr, y1_lr, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
X_train_lr_case3, X_test_lr_case3, y_train_lr_case3, y_test_lr_case3 = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
#决策树回归
X1_dct = SMOTE_resampled_dct.drop(columns=['Label']).values  # 构建特征数据集
y1_dct = SMOTE_resampled_dct['Label'].values  # 构建标签集
X_train_dct, X_test_dct, y_train_dct, y_test_dct = train_test_split(X1_dct, y1_dct, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
X_train_dct_case3, X_test_dct_case3, y_train_dct_case3, y_test_dct_case3 = train_test_split(X_dct, y_dct, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
#随机森林回归
X1_rfc = SMOTE_resampled_rfc.drop(columns=['Label']).values  # 构建特征数据集
y1_rfc = SMOTE_resampled_rfc['Label'].values  # 构建标签集
X_train_rfc, X_test_rfc, y_train_rfc, y_test_rfc = train_test_split(X1_rfc, y1_rfc, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
X_train_rfc_case3, X_test_rfc_case3, y_train_rfc_case3, y_test_rfc_case3 = train_test_split(X_rfc, y_rfc, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
#XGBoost回归
X1_xgbc = SMOTE_resampled_xgbc.drop(columns=['Label']).values  # 构建特征数据集
y1_xgbc = SMOTE_resampled_xgbc['Label'].values  # 构建标签集
X_train_xgbc, X_test_xgbc, y_train_xgbc, y_test_xgbc = train_test_split(X1_xgbc, y1_xgbc, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
X_train_xgbc_case3, X_test_xgbc_case3, y_train_xgbc_case3, y_test_xgbc_case3 = train_test_split(X_xgbc, y_xgbc, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
#LNN回归
X1_lnn = SMOTE_resampled_lnn.drop(columns=['Label']).values  # 构建特征数据集
y1_lnn = SMOTE_resampled_lnn['Label'].values  # 构建标签集
X_train_lnn, X_test_lnn, y_train_lnn, y_test_lnn = train_test_split(X1_lnn, y1_lnn, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集
X_train_lnn_case3, X_test_lnn_case3, y_train_lnn_case3, y_test_lnn_case3 = train_test_split(X_lnn, y_lnn, test_size=0.3, random_state=42) #划分训练集和测试集，20%作为测试集

# 绘制ROC曲线
def plot_roc_curve(fprs, tprs, aucs, labels, title):
    plt.figure(figsize=(10, 8))
    for fpr, tpr, auc, label in zip(fprs, tprs, aucs, labels):
        # 生成一个随机颜色
        color = f'#{random.randint(0, 0xFFFFFF):06x}'
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.4f})',color=color)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    # 保存图像，设置分辨率为300DPI
    plt.savefig('LNN_features_auc_compare.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight'用于确保图像的边界紧凑
    plt.show()

# 构建逻辑回归分类模型
#第一种情况：使用筛选的特征量、使用SMOTE技术
lr = LogisticRegression()  # lr,就代表是逻辑回归模型
lr.fit(X_train_lr, y_train_lr)  # fit,拟合 就相当于是梯度下降
y_pred_lr = lr.predict(X_test_lr)  # 预测
preds_lr = lr.predict_proba(X_test_lr)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test_lr, preds_lr) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_lr = metrics.auc(fpr, tpr)  # 计算AUC值

#第二种情况，使用原始特征量、使用SMOTE技术
lr_case2 = LogisticRegression()  # lr,就代表是逻辑回归模型
lr_case2.fit(X_train, y_train)  # fit,拟合 就相当于是梯度下降
y_pred_lr_case2 = lr_case2.predict(X_test)  # 预测
preds_lr_case2 = lr_case2.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_lr_case2) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_lr_case2 = metrics.auc(fpr, tpr)  # 计算AUC值

#第三种情况：使用筛选的特征量、未使用SMOTE技术
lr_case3 = LogisticRegression()  # lr,就代表是逻辑回归模型
lr_case3.fit(X_train_lr_case3, y_train_lr_case3)  # fit,拟合 就相当于是梯度下降
y_pred_lr_case3 = lr_case3.predict(X_test_lr_case3)  # 预测
preds_lr_case3 = lr_case3.predict_proba(X_test_lr_case3)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test_lr_case3, preds_lr_case3) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_lr_case3 = metrics.auc(fpr, tpr)  # 计算AUC值

# 构建决策树分类模型,手动设置决策树参数
#第一种情况：使用筛选的特征量、使用SMOTE技术
dct = DecisionTreeClassifier(
    max_depth=10,         # 树的最大深度
    min_samples_split=2,   # 分割内部节点所需的最小样本数
    min_samples_leaf=4,    # 叶子节点所需的最小样本数
    criterion='gini',      # 分割质量的测量标准
    random_state=123      # 随机数生成器的种子
)
dct.fit(X_train_dct, y_train_dct)  # 拟合
y_pred_dct = dct.predict(X_test_dct)  # 预测
preds_dct = dct.predict_proba(X_test_dct)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test_dct, preds_dct) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_dct = metrics.auc(fpr, tpr)  # 计算AUC值

#第二种情况，使用原始特征量、使用SMOTE技术
dct_case2 = DecisionTreeClassifier(
    max_depth=10,         # 树的最大深度
    min_samples_split=2,   # 分割内部节点所需的最小样本数
    min_samples_leaf=4,    # 叶子节点所需的最小样本数
    criterion='gini',      # 分割质量的测量标准
    random_state=123      # 随机数生成器的种子
)
dct_case2.fit(X_train, y_train)  # 拟合
y_pred_dct_case2 = dct_case2.predict(X_test)  # 预测
preds_dct_case2 = dct_case2.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_dct_case2) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_dct_case2 = metrics.auc(fpr, tpr)  # 计算AUC值

#第三种情况：使用筛选的特征量、未使用SMOTE技术
dct_case3 = DecisionTreeClassifier(
    max_depth=10,         # 树的最大深度
    min_samples_split=2,   # 分割内部节点所需的最小样本数
    min_samples_leaf=4,    # 叶子节点所需的最小样本数
    criterion='gini',      # 分割质量的测量标准
    random_state=123      # 随机数生成器的种子
)
dct_case3.fit(X_train_dct_case3, y_train_dct_case3)  # 拟合
y_pred_dct_case3 = dct_case3.predict(X_test_dct_case3)  # 预测
preds_dct_case3 = dct_case3.predict_proba(X_test_dct_case3)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test_dct_case3, preds_dct_case3) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_dct_case3 = metrics.auc(fpr, tpr)  # 计算AUC值

# 构建随机森林分类模型，手动设置随机森林参数
#第一种情况：使用筛选的特征量、使用SMOTE技术
rfc = RandomForestClassifier(
    n_estimators=50,  # 树的数量
    max_depth=2,      # 树的最大深度
    min_samples_split=2,  # 分割内部节点所需的最小样本数
    min_samples_leaf=4,  # 叶子节点所需的最小样本数
    max_features='auto',  # 最大特征数
    random_state=42      # 随机数生成器的种子
)
rfc.fit(X_train_rfc, y_train_rfc)  # 拟合
y_pred_rfc = rfc.predict(X_test_rfc)  # 预测
preds_rfc = rfc.predict_proba(X_test_rfc)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test_rfc, preds_rfc) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_rfc = metrics.auc(fpr, tpr)  # 计算AUC值

#第二种情况，使用原始特征量、使用SMOTE技术
rfc_case2 = RandomForestClassifier(
    n_estimators=50,  # 树的数量
    max_depth=2,      # 树的最大深度
    min_samples_split=2,  # 分割内部节点所需的最小样本数
    min_samples_leaf=4,  # 叶子节点所需的最小样本数
    max_features='auto',  # 最大特征数
    random_state=42      # 随机数生成器的种子
)
rfc_case2.fit(X_train, y_train)  # 拟合
y_pred_rfc_case2 = rfc_case2.predict(X_test)  # 预测
preds_rfc_case2 = rfc_case2.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_rfc_case2) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_rfc_case2 = metrics.auc(fpr, tpr)  # 计算AUC值

#第三种情况：使用筛选的特征量、未使用SMOTE技术
rfc_case3 = RandomForestClassifier(
    n_estimators=50,  # 树的数量
    max_depth=2,      # 树的最大深度
    min_samples_split=2,  # 分割内部节点所需的最小样本数
    min_samples_leaf=4,  # 叶子节点所需的最小样本数
    max_features='auto',  # 最大特征数
    random_state=42      # 随机数生成器的种子
)
rfc_case3.fit(X_train_rfc_case3, y_train_rfc_case3)  # 拟合
y_pred_rfc_case3 = rfc_case3.predict(X_test_rfc_case3)  # 预测
preds_rfc_case3 = rfc_case3.predict_proba(X_test_rfc_case3)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test_rfc_case3, preds_rfc_case3) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_rfc_case3 = metrics.auc(fpr, tpr)  # 计算AUC值

# 构建XGBoost分类模型，手动设置XGBoost参数
#第一种情况：使用筛选的特征量、使用SMOTE技术
xgbc = XGBClassifier(
    n_estimators=30,     # 树的数量
    max_depth=1,          # 树的最大深度
    learning_rate=0.1,     # 学习率
    gamma=0,              # 防止过拟合的参数
    subsample=1.0,        # 构建树时用于训练的样本比例
    colsample_bytree=1.0, # 构建树时用于训练的特征比例
    use_label_encoder=False, # 禁用标签编码器
    eval_metric='logloss', # 评估指标
    random_state=42       # 随机数生成器的种子
)
xgbc.fit(X_train_xgbc, y_train_xgbc)  # 拟合
y_pred_xgbc = xgbc.predict(X_test_xgbc)  # 预测
preds_xgbc = xgbc.predict_proba(X_test_xgbc)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test_xgbc, preds_xgbc) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_xgbc = metrics.auc(fpr, tpr)  # 计算AUC值

#第二种情况，使用原始特征量、使用SMOTE技术
xgbc_case2 = XGBClassifier(
    n_estimators=30,     # 树的数量
    max_depth=1,          # 树的最大深度
    learning_rate=0.1,     # 学习率
    gamma=0,              # 防止过拟合的参数
    subsample=1.0,        # 构建树时用于训练的样本比例
    colsample_bytree=1.0, # 构建树时用于训练的特征比例
    use_label_encoder=False, # 禁用标签编码器
    eval_metric='logloss', # 评估指标
    random_state=42       # 随机数生成器的种子
)
xgbc_case2.fit(X_train, y_train)  # 拟合
y_pred_xgbc_case2 = xgbc_case2.predict(X_test)  # 预测
preds_xgbc_case2 = xgbc_case2.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds_xgbc_case2) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_xgbc_case2 = metrics.auc(fpr, tpr)  # 计算AUC值

#第三种情况：使用筛选的特征量、未使用SMOTE技术
xgbc_case3 = XGBClassifier(
    n_estimators=30,     # 树的数量
    max_depth=1,          # 树的最大深度
    learning_rate=0.1,     # 学习率
    gamma=0,              # 防止过拟合的参数
    subsample=1.0,        # 构建树时用于训练的样本比例
    colsample_bytree=1.0, # 构建树时用于训练的特征比例
    use_label_encoder=False, # 禁用标签编码器
    eval_metric='logloss', # 评估指标
    random_state=42       # 随机数生成器的种子
)
xgbc_case3.fit(X_train_xgbc_case3, y_train_xgbc_case3)  # 拟合
y_pred_xgbc_case3 = xgbc_case3.predict(X_test_xgbc_case3)  # 预测
preds_xgbc_case3 = xgbc_case3.predict_proba(X_test_xgbc_case3)[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test_xgbc_case3, preds_xgbc_case3) # 该函数返回这三个变量：fpr,tpr,和阈值thresholds;
roc_auc_xgbc_case3 = metrics.auc(fpr, tpr)  # 计算AUC值

#compare the four model
#for model_name, y_pred, y_test, probs_model in [('Logistic Regression', y_pred_lr, y_test_lr, preds_lr),
#                                               ('Decision Tree', y_pred_dct, y_test_dct, preds_dct),
#                                               ('Random Forest', y_pred_rfc, y_test_rfc, preds_rfc),
#                                               ('XGBoost', y_pred_xgbc, y_test_xgbc, preds_xgbc),
#                                               ('LNN', y_pred_lnn, y_test_lnn, preds_lnn),
#                                              ]:
#    fpr, tpr, _ = roc_curve(y_test, probs_model)
#    auc = metrics.auc(fpr, tpr)
#    fprs.append(fpr)
#    tprs.append(tpr)
#    aucs.append(auc)
#    labels.append(model_name)

#plot_roc_curve(fprs, tprs, aucs, labels, 'ROC Curve Comparison')

#for model_name, y_pred, y_test, probs_model in [('LR_with Optimal Features', y_pred_lr, y_test_lr, preds_lr),
#                                               ('LR_with all Features', y_pred_lr_case2, y_test, preds_lr_case2),
#                                              ]:
#    fpr, tpr, _ = roc_curve(y_test, probs_model)
#    auc = metrics.auc(fpr, tpr)
#    fprs.append(fpr)
#    tprs.append(tpr)
#    aucs.append(auc)
#    labels.append(model_name)

#plot_roc_curve(fprs, tprs, aucs, labels, 'LR:ROC Curve Comparison')

#for model_name, y_pred, y_test, probs_model in [('DCT_with Optimal Features', y_pred_dct, y_test_dct, preds_dct),
#                                               ('DCT_with all Features', y_pred_dct_case2, y_test, preds_dct_case2),
#                                              ]:
#    fpr, tpr, _ = roc_curve(y_test, probs_model)
#    auc = metrics.auc(fpr, tpr)
#    fprs.append(fpr)
#    tprs.append(tpr)
#    aucs.append(auc)
#    labels.append(model_name)

#plot_roc_curve(fprs, tprs, aucs, labels, 'DCT:ROC Curve Comparison')

#for model_name, y_pred, y_test, probs_model in [('RF_with Optimal Features', y_pred_rfc, y_test_rfc, preds_rfc),
#                                               ('RF_with all Features', y_pred_rfc_case2, y_test, preds_rfc_case2),
#                                              ]:
#    fpr, tpr, _ = roc_curve(y_test, probs_model)
#    auc = metrics.auc(fpr, tpr)
#    fprs.append(fpr)
#    tprs.append(tpr)
#    aucs.append(auc)
#    labels.append(model_name)

#plot_roc_curve(fprs, tprs, aucs, labels, 'RF:ROC Curve Comparison')

#for model_name, y_pred, y_test, probs_model in [('XGBoost_with Optimal Features', y_pred_xgbc_case2, y_test, preds_xgbc_case2),
#                                                ('XGBoost_with all Features', y_pred_xgbc, y_test_xgbc, preds_xgbc),
#                                              ]:
#    fpr, tpr, _ = roc_curve(y_test, probs_model)
#    auc = metrics.auc(fpr, tpr)
#    fprs.append(fpr)
#    tprs.append(tpr)
#    aucs.append(auc)
#    labels.append(model_name)

#plot_roc_curve(fprs, tprs, aucs, labels, 'XGBoost:ROC Curve Comparison')
