import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from joblib import dump, load
from torch.utils.tensorboard import SummaryWriter

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据
file_path = "Secure_Cyber_Systems/data/final_dataset_with_features.csv"
data = pd.read_csv(file_path)

# 提取URL和标签
urls = data['url']
y = data['label']

# 对标签进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 创建字符到索引的映射
char_to_idx = {char: idx + 1 for idx, char in enumerate(set(''.join(urls)))}
max_length = 100  # 可以根据数据调整

# 将URL转换为索引序列
def url_to_sequence(url):
    return [char_to_idx.get(char, 0) for char in url]

# 将URL转换为序列并填充
X_seq = [url_to_sequence(url) for url in urls]
X_padded = []
for seq in X_seq:
    if len(seq) < max_length:
        padded_seq = seq + [0] * (max_length - len(seq))
    else:
        padded_seq = seq[:max_length]
    X_padded.append(padded_seq)

X_padded = np.array(X_padded)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 创建Dataset和DataLoader
class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = URLDataset(X_train_tensor, y_train_tensor)
test_dataset = URLDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义结合LSTM和Transformer的模型
class LSTMTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, transformer_hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(LSTMTransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=lstm_hidden_dim * 2, nhead=8, dim_feedforward=transformer_hidden_dim, dropout=dropout),
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)
        transformer_output = self.transformer(lstm_output.permute(1, 0, 2))
        hidden = transformer_output.mean(dim=0)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# 参数设置
vocab_size = len(char_to_idx) + 1  # 词汇表大小
embedding_dim = 128  # 嵌入维度
lstm_hidden_dim = 128  # LSTM隐藏层维度
transformer_hidden_dim = 256  # Transformer隐藏层维度
output_dim = 1  # 输出维度（二分类）
num_layers = 2  # LSTM和Transformer层数
dropout = 0.5  # Dropout率

# 将模型移动到设备（GPU或CPU）
model = LSTMTransformerClassifier(vocab_size, embedding_dim, lstm_hidden_dim, transformer_hidden_dim, output_dim, num_layers=num_layers, dropout=dropout).to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# 使用 TensorBoard
log_dir = 'Secure_Cyber_Systems/lstm_train_transformer/logs'
writer = SummaryWriter(log_dir=log_dir)

# 训练模型
num_epochs = 20
early_stopping_patience = 3
best_loss = float('inf')
counter = 0

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        # 将数据移动到设备
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X).squeeze(1)
        loss = criterion(predictions, batch_y.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # 记录训练损失
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    writer.add_scalar('Training Loss', avg_train_loss, epoch)
    
    print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")
    scheduler.step(avg_train_loss)

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # 将数据移动到设备
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X).squeeze(1)
            loss = criterion(predictions, batch_y.float())
            val_loss += loss.item()
            predicted = (torch.sigmoid(predictions) > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y.float()).sum().item()
    
    # 记录验证损失和准确率
    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    accuracy = correct / total
    val_accuracies.append(accuracy)
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation Accuracy', accuracy, epoch)
    
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 早停法
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

# 关闭 TensorBoard writer
writer.close()

# 保存模型为 .joblib 文件
# model.cpu()
dump(model, 'Secure_Cyber_Systems/lstm_train_transformer/phishing_detector_lstm_transformer.joblib')
print("Model saved as phishing_detector_lstm_transformer.joblib")

# 绘制训练和验证的 Loss 曲线
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Secure_Cyber_Systems/lstm_train_transformer/loss_curve.png')
plt.show()

# 绘制验证的 Accuracy 曲线
plt.figure()
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig('Secure_Cyber_Systems/lstm_train_transformer/accuracy_curve.png')
plt.show()

# 模型评估
model.eval()
all_predictions = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        predictions = model(batch_X).squeeze(1)
        predicted = (torch.sigmoid(predictions) > 0.5).float()
        probs = torch.sigmoid(predictions)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(cm)

# 精确率和召回率
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# F1 分数
f1 = f1_score(all_labels, all_predictions)
print(f"F1 Score: {f1:.4f}")

# ROC 曲线和 AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)

# 使用 TensorBoard 记录 ROC 曲线
writer = SummaryWriter(log_dir=log_dir)
writer.add_pr_curve('ROC Curve', torch.tensor(all_labels), torch.tensor(all_probs), global_step=epoch)
writer.close()

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('Secure_Cyber_Systems/lstm_train_transformer/roc_curve.png')
plt.show()