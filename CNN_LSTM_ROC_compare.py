# 导入第三方库
import matplotlib.pyplot as plt
import warnings, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_curve, auc)
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import keras.layers as layers
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
import numpy as np

warnings.filterwarnings(action='ignore')

# 读取数据
df = pd.read_excel('test_data.xlsx')

# 数据可视化
plt.figure(figsize=(20, 16))
sns.heatmap(df.iloc[:, :].corr().round(2), cmap="YlGnBu", annot=True, annot_kws={"size": 4})
plt.title('Heat map analysis results')
plt.xticks(fontsize=4)
plt.yticks(fontsize=4)
plt.show()

# ================== 特征工程 ==================
common_features = [
    "HBDH", "CRP", "NEU#", "WBC", "INR", "CREA", "CA", "UREA", "FIB",
    "AMY", "EOS%", "HGB", "HDL-C", "A/G", "BAS%", "TG", "APTT", "RDW-SD",
    "RBC", "CHE", "LDL-C", "RDW-CV", "MON%", "MCH", "PT", "DBIL", "CO2-L"
]

models_config = {
    'LSTM': {'X': df[common_features], 'y': df.Label},
    'CNN': {'X': df[common_features], 'y': df.Label}
}

# 数据均衡化处理
smote = SMOTE(random_state=0)
for model_name in models_config:
    X_res, y_res = smote.fit_resample(models_config[model_name]['X'], models_config[model_name]['y'])
    models_config[model_name]['X_res'] = pd.DataFrame(X_res, columns=common_features)
    models_config[model_name]['y_res'] = pd.Series(y_res, name='Label')

# 数据集划分
for model_name in models_config:
    X_train, X_test, y_train, y_test = train_test_split(
        models_config[model_name]['X_res'].values,
        models_config[model_name]['y_res'].values,
        test_size=0.3,
        random_state=42
    )
    models_config[model_name].update({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    })

# ================== 模型构建 ==================
def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

# LSTM模型
lstm = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(models_config['LSTM']['X_train'].shape[1], 1)),
    LSTM(50),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# CNN模型
cnn = Sequential([
    Conv1D(3, 4, activation='relu',
           input_shape=(models_config['CNN']['X_train'].shape[1], 1)),
    MaxPooling1D(strides=1, padding='same'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# ================== 模型训练 ==================
# 调整数据维度
X_train_lstm = K.expand_dims(models_config['LSTM']['X_train'], -1)
X_test_lstm = K.expand_dims(models_config['LSTM']['X_test'], -1)

X_train_cnn = K.expand_dims(models_config['CNN']['X_train'], -1)
X_test_cnn = K.expand_dims(models_config['CNN']['X_test'], -1)

history_lstm = lstm.fit(
    X_train_lstm, models_config['LSTM']['y_train'],
    validation_data=(X_test_lstm, models_config['LSTM']['y_test']),
    epochs=200,
    batch_size=64,
    verbose=0
)

history_cnn = cnn.fit(
    X_train_cnn, models_config['CNN']['y_train'],
    validation_data=(X_test_cnn, models_config['CNN']['y_test']),
    epochs=50,
    batch_size=64,
    verbose=0
)

# ================== 模型评估 ==================
def evaluate_model(model, config):
    X_test = K.expand_dims(config['X_test'], -1)
    y_pred = np.round(model.predict(X_test))
    probs = model.predict(X_test).ravel()

    return {
        'accuracy': accuracy_score(config['y_test'], y_pred),
        'precision': precision_score(config['y_test'], y_pred),
        'recall': recall_score(config['y_test'], y_pred),
        'f1': f1_score(config['y_test'], y_pred),
        'specificity': calculate_specificity(config['y_test'], y_pred),
        'fpr': roc_curve(config['y_test'], probs)[0],
        'tpr': roc_curve(config['y_test'], probs)[1],
        'auc': auc(roc_curve(config['y_test'], probs)[0],
                   roc_curve(config['y_test'], probs)[1])
    }

results = {
    'LSTM': evaluate_model(lstm, models_config['LSTM']),
    'CNN': evaluate_model(cnn, models_config['CNN'])
}

# ================== 结果可视化 ==================
def plot_training_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'r', label='Training loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation loss')
    plt.title(f'{title} Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], 'r', label='Training accuracy')
    plt.plot(history.history['val_acc'], 'b', label='Validation accuracy')
    plt.title(f'{title} Accuracy')
    plt.legend()
    plt.show()

plot_training_history(history_lstm, 'LSTM')
plot_training_history(history_cnn, 'CNN')

# 绘制ROC曲线对比
plt.figure(figsize=(10, 8))
for model_name in ['LSTM', 'CNN']:
    plt.plot(results[model_name]['fpr'],
             results[model_name]['tpr'],
             label=f'{model_name} (AUC = {results[model_name]["auc"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve comparison of LSTM and CNN')
plt.legend(loc="lower right")
plt.savefig('ROC_Comparison_CNN_LSTM.png', dpi=300, bbox_inches='tight')
plt.show()