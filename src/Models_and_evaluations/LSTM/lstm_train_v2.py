import pandas as pd
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.impute import SimpleImputer
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义增强型LSTM模型
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(2*hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context).squeeze()

# 定义数据集类
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 模型管道保存/加载
def save_pipeline(model, preprocessor, filepath):
    """保存完整模型管道"""
    model_copy = copy.deepcopy(model).cpu()
    pipeline = {
        'model_state': model_copy.state_dict(),
        'model_meta': {
            'input_dim': model_copy.lstm.input_size,
            'hidden_dim': model_copy.lstm.hidden_size,
            'num_layers': model_copy.lstm.num_layers,
            'dropout': model_copy.lstm.dropout
        },
        'preprocessor': preprocessor
    }
    dump(pipeline, filepath)
    print(f"Pipeline saved to {filepath}")

def load_pipeline(filepath, device='cpu'):
    """加载完整模型管道"""
    pipeline = load(filepath)
    
    model = AttentionLSTM(
        input_dim=pipeline['model_meta']['input_dim'],
        hidden_dim=pipeline['model_meta']['hidden_dim'],
        num_layers=pipeline['model_meta']['num_layers'],
        dropout=pipeline['model_meta']['dropout']
    ).to(device)
    
    model.load_state_dict(pipeline['model_state'])
    model.eval()
    
    return model, pipeline['preprocessor']

# 超参数优化函数
def objective(trial, X_scaled, y_encoded):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 3),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }
    
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y_encoded)):
        train_loader = DataLoader(FeatureDataset(X_scaled[train_idx], y_encoded[train_idx]), 
                                batch_size=128, shuffle=True)
        val_loader = DataLoader(FeatureDataset(X_scaled[val_idx], y_encoded[val_idx]), 
                              batch_size=256)
        
        # 初始化模型
        model = AttentionLSTM(
            input_dim=X_scaled.shape[1],
            **{k:v for k,v in params.items() if k in ['hidden_dim', 'num_layers', 'dropout']}
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), 
                              lr=params['lr'],
                              weight_decay=params['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        criterion = nn.BCEWithLogitsLoss()
        
        # 快速训练
        best_fold_auc = 0
        for epoch in range(10):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # 验证
            model.eval()
            probs, labels = [], []
            with torch.no_grad():
                for inputs, batch_labels in val_loader:
                    probs.extend(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
                    labels.extend(batch_labels.numpy())
            
            fold_auc = roc_auc_score(labels, probs)
            trial.report(fold_auc, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            best_fold_auc = max(best_fold_auc, fold_auc)
        
        fold_results.append(best_fold_auc)
    
    return np.mean(fold_results)

# 主训练函数
def train_with_best_params(X_scaled, y_encoded, best_params):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    histories = []
    preprocessor = {'imputer': imputer, 'scaler': scaler, 'encoder': le}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y_encoded)):
        print(f"\n=== Training Fold {fold+1} ===")
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
        
        train_loader = DataLoader(FeatureDataset(X_scaled[train_idx], y_encoded[train_idx]),
                                batch_size=64, shuffle=True)
        val_loader = DataLoader(FeatureDataset(X_scaled[val_idx], y_encoded[val_idx]),
                              batch_size=128)
        
        # 使用最佳参数初始化模型
        model = AttentionLSTM(
            input_dim=X_scaled.shape[1],
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(),
                              lr=best_params['lr'],
                              weight_decay=best_params['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=20)
        criterion = nn.BCEWithLogitsLoss()
        
        best_auc = 0
        early_stop_counter = 0
        
        for epoch in range(1, 31):

            model.train()
            train_loss = 0
            
            # 训练步骤
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证步骤
            model.eval()
            val_loss, correct, total = 0, 0, 0
            all_probs, all_labels = [], []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs = model(inputs).squeeze()
                    val_loss += criterion(outputs, labels).item()
                    
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # 计算指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            val_auc = roc_auc_score(all_labels, all_probs)
            
            # 记录结果
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # 打印进度
            print(f"Epoch {epoch+1}/15 | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"AUC: {val_auc:.4f}")
            
            # 保存最佳模型
            if val_auc > best_auc:
                save_pipeline(model, preprocessor, f"best_fold{fold}.joblib")
                best_auc = val_auc
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= 5:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            histories.append(history)
    
    return histories

# 主程序
if __name__ == "__main__":
    # 数据加载与预处理
    data = pd.read_csv("Secure_Cyber_Systems/data/final_dataset_with_features.csv")
    feature_columns = data.columns.drop(['url', 'label'])
    X = data[feature_columns]
    y = data['label']
    
    # 预处理管道
    imputer = SimpleImputer(strategy='median').fit(X)
    X = pd.DataFrame(imputer.transform(X), columns=feature_columns)
    le = LabelEncoder().fit(y)
    y_encoded = le.transform(y)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # 超参数搜索阶段
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(trial, X_scaled, y_encoded), n_trials=30)
    
    print(f"\nBest Params: {study.best_params}")
    print(f"Best AUC: {study.best_value:.4f}")
    
    # 使用最佳参数进行最终训练
    histories = train_with_best_params(X_scaled, y_encoded, study.best_params)
    
    # 可视化结果
    plt.figure(figsize=(15,5))
    for i, metric in enumerate(['loss', 'acc', 'auc']):
        plt.subplot(1,3,i+1)
        for fold, hist in enumerate(histories):
            plt.plot(hist[f'val_{metric}'], label=f'Fold {fold+1}')
        plt.title(f'Validation {metric.upper()}')
        plt.xlabel('Epoch')
        plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')