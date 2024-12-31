import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, accuracy_score, precision_score, recall_score)
from imblearn.over_sampling import RandomOverSampler, SMOTE  #导入过采样工具

# Sıvı Sinir Ağı Modeli
class LiquidNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, time_constant=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_in = np.random.randn(input_size, hidden_size)
        self.W_hid = np.random.randn(hidden_size, hidden_size)
        self.W_out = np.random.randn(hidden_size, output_size)
        self.bias_hid = np.zeros(hidden_size)
        self.bias_out = np.zeros(output_size)
        self.time_constant = time_constant

    def forward(self, x):
        hidden_state = np.zeros(self.hidden_size)
        outputs = []

        for t in range(len(x)):
            input_flattened = x[t].reshape(-1)
            hidden_state = (1 - self.time_constant) * hidden_state + \
                            self.time_constant * (np.dot(input_flattened, self.W_in) + self.bias_hid) + \
                            np.dot(hidden_state, self.W_hid)
            output = np.dot(hidden_state, self.W_out) + self.bias_out
            exp_output = np.exp(output - np.max(output))  # Numerik kararlılık için max çıkarıldı
            softmax_output = exp_output / np.sum(exp_output)
            outputs.append(softmax_output)

        return np.array(outputs), hidden_state

    def train(self, x_train, y_train, x_val, y_val, epochs=300, learning_rate=0.01):
        y_train_one_hot = self.one_hot_encode(y_train, self.output_size)
        y_val_one_hot = self.one_hot_encode(y_val, self.output_size)
        self.train_losses = []
        self.val_losses = []

        for epoch in range(epochs):
            total_loss = 0
            total_val_loss = 0

            # Eğitim kaybı için
            for i in range(len(x_train)):
                predictions, hidden_state = self.forward(x_train[i:i+1])
                prediction = predictions[-1]
                loss = -np.sum(y_train_one_hot[i] * np.log(prediction + 1e-9))
                total_loss += loss
                d_output = prediction - y_train_one_hot[i]
                d_hidden = np.dot(d_output, self.W_out.T) * self.time_constant
                self.W_out -= learning_rate * np.outer(hidden_state, d_output)
                self.bias_out -= learning_rate * d_output
                self.W_in -= learning_rate * np.outer(x_train[i].reshape(-1), d_hidden)
                self.W_hid -= learning_rate * np.outer(hidden_state, d_hidden)
                self.bias_hid -= learning_rate * d_hidden

            avg_loss = total_loss / len(x_train)
            self.train_losses.append(avg_loss)

            # Doğrulama kaybı için
            for i in range(len(x_val)):
                val_predictions, _ = self.forward(x_val[i:i+1])
                val_prediction = val_predictions[-1]
                val_loss = -np.sum(y_val_one_hot[i] * np.log(val_prediction + 1e-9))
                total_val_loss += val_loss

            avg_val_loss = total_val_loss / len(x_val)
            self.val_losses.append(avg_val_loss)

            train_accuracy, _ = self.evaluate(x_train, y_train)
            val_accuracy, _ = self.evaluate(x_val, y_val)

            print(f"Epoch {epoch + 1}/{epochs}, Eğitim Kayıp: {avg_loss:.4f}, Eğitim Doğruluğu: {train_accuracy:.4f}, Doğrulama Kayıp: {avg_val_loss:.4f}, Doğrulama Doğruluğu: {val_accuracy:.4f}")

    def evaluate(self, x, y):
        predictions = [self.forward(x[i:i+1])[0][-1] for i in range(len(x))]
        predicted_labels = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(y, predicted_labels)
        return accuracy, np.array(predictions)

    def one_hot_encode(self, y, num_classes):
        return np.eye(num_classes)[y]

    def plot_loss(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Eğitim Kayıp', color='blue')
        plt.plot(self.val_losses, label='Doğrulama Kayıp', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.title('Eğitim ve Doğrulama Kayıp Eğrisi')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, title='Karışıklık Matrisi'):
        conf_matrix = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=[str(i) for i in np.unique(y_true)])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.show()

    def plot_predictions(self, x, y_true, y_pred, num_images=10):
        plt.figure(figsize=(15, 6))
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1)
            plt.bar(range(self.output_size), y_pred[i], color='blue')
            plt.ylim(0, 1)
            plt.xlabel('Sınıf')
            plt.ylabel('Olasılık')
            plt.title('Tahmin Olasılıkları')

            predicted_label = np.argmax(y_pred[i])
            true_label = y_true.iloc[i]

            color = 'green' if predicted_label == true_label else 'red'
            plt.subplot(3, num_images, num_images + i + 1)
            plt.bar(range(self.output_size), y_pred[i], color=color)
            plt.ylim(0, 1)
            plt.xlabel('Sınıf')
            plt.ylabel('Olasılık')
            plt.title(f"Gerçek: {true_label}\nTahmin: {predicted_label}", color=color)

        plt.tight_layout()
        plt.show()

# 读取数据
df = pd.read_excel('test_data.xlsx')

# 提取特征变量和标签变量
y = df.Label
X = df.drop('Label', axis=1)

#数据均衡化
model_SMOTE = SMOTE(random_state=0)  # 建立过采样模型
x_SMOTE_resampled, y_SMOTE_resampled = model_SMOTE.fit_resample(X, y)  # 过采样后样本结果
x_SMOTE_resampled = pd.DataFrame(x_SMOTE_resampled, columns=X.columns.values)  # 构建DataFrame框架
y_SMOTE_resampled = pd.DataFrame(y_SMOTE_resampled, columns=['Label'])  # 构建DataFrame框架
SMOTE_resampled = pd.concat([x_SMOTE_resampled, y_SMOTE_resampled], axis=1)  # 框架合并
groupby_data_SMOTE = SMOTE_resampled.groupby('Label').count()  # 分组统计数据
print(groupby_data_SMOTE)  # 查看分组的数据

#利用过采样后的样本来进行训练
X1 = SMOTE_resampled.drop(columns=['Label']).values  # 构建特征数据集
y1 = SMOTE_resampled['Label'].values  # 构建标签集

# Eğitim ve test setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Validation set oluşturmak için
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

input_size = x_train.shape[1]
hidden_size = 100  # Gizli katman boyutunu artırdık
output_size = 1
time_constant = 0.1  # Zaman sabiti
net = LiquidNeuralNetwork(input_size, hidden_size, output_size, time_constant)

net.train(x_train, y_train, x_val, y_val, epochs=50, learning_rate=0.005)  # Öğrenme oranını azalttık

# Sonuçları değerlendirme
train_accuracy, y_pred_train = net.evaluate(x_train, y_train)
test_accuracy, y_pred_test = net.evaluate(x_test, y_test)

print(f"Eğitim doğruluğu: {train_accuracy * 100:.2f}%")
print(f"Test doğruluğu: {test_accuracy * 100:.2f}%")

net.plot_loss()

# Karışıklık matrisini oluşturmak için doğru biçimlendirme
y_pred_train_labels = np.argmax(y_pred_train, axis=1)  # Model tahminlerini sınıf etiketlerine dönüştür
y_pred_test_labels = np.argmax(y_pred_test, axis=1)

# Eğitim verileri için karışıklık matrisi
net.plot_confusion_matrix(y_train, y_pred_train_labels, 'Eğitim Verileri Karışıklık Matrisi')

# Test verileri için karışıklık matrisi
net.plot_confusion_matrix(y_test, y_pred_test_labels, 'Test Verileri Karışıklık Matrisi')

# Performans metriklerini hesapla ve yazdır
conf_matrix = confusion_matrix(y_test, y_pred_test_labels)
print("Confusion Matrix:\n", conf_matrix)
print("F1 Score:", f1_score(y_test, y_pred_test_labels))
print("Accuracy Score:", accuracy_score(y_test, y_pred_test_labels))
print("Precision Score:", precision_score(y_test, y_pred_test_labels))
print("Recall Score:", recall_score(y_test, y_pred_test_labels))