import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from joblib import dump, load
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib
import matplotlib.pyplot as plt

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

# 转换为PyTorch张量
X_tensor = torch.LongTensor(X_padded)
y_tensor = torch.LongTensor(y_encoded)

# 创建Dataset和DataLoader
class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(hidden)

# 参数设置
vocab_size = len(char_to_idx) + 1  # 词汇表大小
embedding_dim = 128  # 嵌入维度
hidden_dim = 128  # LSTM隐藏层维度
output_dim = 1  # 输出维度（二分类）
num_layers = 2  # LSTM层数
bidirectional = True  # 双向LSTM
dropout = 0.5  # Dropout率
batch_size = 32  # 批量大小
num_epochs = 20  # 训练轮数
k_folds = 5  # K折交叉验证的折数
early_stopping_patience = 3  # 早停法的耐心值

# 创建K折交叉验证的分组
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# 存储每个折叠的结果
results = {
    'train_losses': [],
    'val_losses': [],
    'val_accuracies': [],
    'precisions': [],
    'recalls': [],
    'f1_scores': [],
    'aucs': []
}

# 执行K折交叉验证
for fold, (train_idx, val_idx) in enumerate(skf.split(X_tensor, y_tensor)):
    print(f"Fold {fold + 1}/{k_folds}")
    
    # 创建训练集和验证集
    X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
    
    # 创建Dataset和DataLoader
    train_dataset = URLDataset(X_train, y_train)
    val_dataset = URLDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 
                          num_layers=num_layers, bidirectional=bidirectional, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    
    # 使用 TensorBoard
    log_dir = f'Secure_Cyber_Systems/lstm_train/logs/fold_{fold}'
    writer = SummaryWriter(log_dir=log_dir)
    
    # 训练模型
    best_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
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
        
        print(f"Fold {fold + 1}, Epoch {epoch+1}, Loss: {avg_train_loss}")
        scheduler.step(avg_train_loss)
        
        # 验证模型
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_X).squeeze(1)
                loss = criterion(predictions, batch_y.float())
                val_loss += loss.item()
                predicted = (torch.sigmoid(predictions) > 0.5).float()
                probs = torch.sigmoid(predictions)
                
                total += batch_y.size(0)
                correct += (predicted == batch_y.float()).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 记录验证损失和准确率
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation Accuracy', accuracy, epoch)
        
        print(f"Fold {fold + 1}, Epoch {epoch+1}, Accuracy: {accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 早停法
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= early_stopping_patience:
                print("Early stopping triggered")
                break
    
    # 记录结果
    results['train_losses'].append(train_losses)
    results['val_losses'].append(val_losses)
    results['val_accuracies'].append(val_accuracies)
    
    # 模型评估
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    
    results['precisions'].append(precision)
    results['recalls'].append(recall)
    results['f1_scores'].append(f1)
    results['aucs'].append(auc)
    
    # 关闭 TensorBoard writer
    writer.close()
    
    # 保存模型
    model_dir = f'Secure_Cyber_Systems/lstm_train/models/fold_{fold}'
    os.makedirs(model_dir, exist_ok=True)
    dump(model, f'{model_dir}/phishing_detector_lstm.joblib')
    print(f"Model for fold {fold + 1} saved as {model_dir}/phishing_detector_lstm.joblib")
    
    # 保存字符到索引的映射
    dump(char_to_idx, f'{model_dir}/char_to_idx.joblib')
    print(f"Character to index mapping for fold {fold + 1} saved as {model_dir}/char_to_idx.joblib")

# 绘制训练和验证的 Loss 曲线（所有折叠的平均值）
plt.figure(figsize=(10, 6))
for fold in range(k_folds):
    plt.plot(results['train_losses'][fold], label=f'Fold {fold + 1} Training Loss')
    plt.plot(results['val_losses'][fold], label=f'Fold {fold + 1} Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Across Folds')
plt.legend()
plt.grid(True)
plt.ylim(0, max(max(loss) for loss in results['train_losses']) * 1.1)  # 调整纵轴范围
plt.yticks(np.arange(0, max(max(loss) for loss in results['train_losses']) * 1.1, 0.1))  # 调整纵轴间隔
plt.savefig('Secure_Cyber_Systems/lstm_train/loss_curve_all_folds.png')
plt.show()

# 绘制验证的 Accuracy 曲线（所有折叠的平均值）
plt.figure(figsize=(10, 6))
for fold in range(k_folds):
    plt.plot(results['val_accuracies'][fold], label=f'Fold {fold + 1} Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Across Folds')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.0)  # 调整纵轴范围
plt.yticks(np.arange(0, 1.1, 0.1))  # 调整纵轴间隔
plt.savefig('Secure_Cyber_Systems/lstm_train/accuracy_curve_all_folds.png')
plt.show()

# 打印平均评估指标
print(f"Average Precision: {np.mean(results['precisions']):.4f}")
print(f"Average Recall: {np.mean(results['recalls']):.4f}")
print(f"Average F1 Score: {np.mean(results['f1_scores']):.4f}")
print(f"Average AUC: {np.mean(results['aucs']):.4f}")