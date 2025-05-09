import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 导入CSV文件
features_df = pd.read_csv('output_feature_data.csv', header=None)
labels_df = pd.read_csv('output_label_data.csv', header=None)

print(features_df.shape)


# 数据预处理
def preprocess_data(features_df, labels_df, num_splits):
    feature_matrices = []
    label_matrices = []

    for i in range(20):
        feature_matrices.append(features_df.iloc[i::20].reset_index(drop=True))
        label_matrices.append(labels_df.iloc[i::20].reset_index(drop=True))

    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

    random_seed = 42
    for i in range(num_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrices[i], label_matrices[i], test_size=0.2, random_state=random_seed)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    X_train = pd.concat(X_train_list).reset_index(drop=True)
    X_test = pd.concat(X_test_list).reset_index(drop=True)
    y_train = pd.concat(y_train_list).reset_index(drop=True)
    y_test = pd.concat(y_test_list).reset_index(drop=True)

    return X_train, X_test, y_train, y_test


# 定义XLSTM层
class XLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.3):
        super(XLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        dropout_out = self.dropout(lstm_out)
        return dropout_out


# 定义局部注意力机制
class LocalAttention(nn.Module):
    def __init__(self, hidden_dim, max_length):
        super(LocalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=max_length, padding=max_length // 2, bias=False)

    def forward(self, Q, K, V):
        # 将 Q, K, V 转换为适合卷积操作的形状
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 使用卷积操作计算局部注意力分数
        scores = self.conv(Q) * K
        A = torch.softmax(scores, dim=-1)

        # 计算加权求和值
        Z = A * V
        Z = Z.transpose(1, 2)

        return Z


# 定义带有局部注意力机制的复杂模型
class ComplexXLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3, max_length=8):
        super(ComplexXLSTM, self).__init__()
        self.xlstm1 = XLSTM(input_dim, hidden_dim, dropout_rate)
        self.xlstm2 = XLSTM(hidden_dim, hidden_dim, dropout_rate)
        self.attention1 = LocalAttention(hidden_dim, max_length)

        self.xlstm3 = XLSTM(hidden_dim, hidden_dim, dropout_rate)
        self.xlstm4 = XLSTM(hidden_dim, hidden_dim, dropout_rate)
        self.attention2 = LocalAttention(hidden_dim, max_length)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.xlstm1(x)
        x = self.xlstm2(x)
        attn1 = self.attention1(x, x, x)
        x = x + attn1  # 使用残差连接

        x = self.xlstm3(x)
        x = self.xlstm4(x)
        attn2 = self.attention2(x, x, x)
        x = x + attn2  # 使用残差连接

        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


input_dim = 1
hidden_dim = 32
output_dim = 1
dropout_rate = 0.3


# 创建模型
def create_model():
    model = ComplexXLSTM(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


criterion = nn.MSELoss()


# 创建数据加载器
def create_dataloader(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# 训练和评估模型
epochs_list = [50, 100, 150, 200, 500, 1000]
batch_sizes = [128, 64, 32, 16, 8]

for num_splits in [9, 10, 11, 12, 13, 14, 15, 16, 17]:
    results = []
    predictions_list = []
    y_tests_list = []

    X_train, X_test, y_train, y_test = preprocess_data(features_df, labels_df, num_splits)

    print(f'Training features shape: {X_train.shape}')
    print(f'Testing features shape: {X_test.shape}')
    print(f'Training labels shape: {y_train.shape}')
    print(f'Testing labels shape: {y_test.shape}')

    # 重塑数据以适应LSTM模型
    X_train = np.expand_dims(X_train.values, axis=-1)
    X_test = np.expand_dims(X_test.values, axis=-1)
    y_train = np.expand_dims(y_train.values, axis=-1)
    y_test = np.expand_dims(y_test.values, axis=-1)

    # 将数据转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    for epochs in epochs_list:
        for batch_size in batch_sizes:
            print(f'Training with epochs={epochs} and batch_size={batch_size}')
            model, optimizer = create_model()
            train_loader = create_dataloader(X_train, y_train, batch_size)

            # 训练模型
            model.train()
            for epoch in range(epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()

            # 评估模型
            model.eval()
            with torch.no_grad():
                y_pred = []
                for X_batch, _ in create_dataloader(X_test, y_test, batch_size, shuffle=False):
                    output = model(X_batch)
                    y_pred.append(output.cpu().numpy())
                y_pred = np.concatenate(y_pred, axis=0)
                predictions_list.append(y_pred)
                y_tests_list.append(y_test.cpu().numpy())

                mse = mean_squared_error(y_test.cpu().numpy().flatten(), y_pred.flatten())
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test.cpu().numpy().flatten(), y_pred.flatten())

                print(f'Epochs: {epochs}, Batch Size: {batch_size}, RMSE: {rmse}, R^2: {r2}')
                results.append((epochs, batch_size, rmse, r2))

    # 保存预测性能参数到CSV
    results_df = pd.DataFrame(results, columns=['Epochs', 'Batch Size', 'RMSE', 'R^2'])
    results_df.to_csv(f'model_performance_metrics_{num_splits}.csv', index=False)

    # 保存所有 y_pred 和 y_test 到CSV文件
    predictions_2d = np.concatenate(predictions_list, axis=0).reshape(-1, predictions_list[0].shape[1])
    predictions_df = pd.DataFrame(predictions_2d)
    predictions_df.to_csv(f'predictions_{num_splits}.csv', index=False)

    y_tests_2d = np.concatenate(y_tests_list, axis=0).reshape(-1, y_tests_list[0].shape[1])
    y_tests_df = pd.DataFrame(y_tests_2d)
    y_tests_df.to_csv(f'y_tests_{num_splits}.csv', index=False)
