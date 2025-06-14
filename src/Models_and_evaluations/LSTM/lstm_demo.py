import torch
from joblib import load
import numpy as np
import socket
import re
from urllib.parse import urlparse, urlunparse

# 设置socket超时（与训练代码一致）
socket.setdefaulttimeout(5)

# 域名解析函数（与训练代码完全一致）
def resolve_domain(url):
    """将域名转换为IP地址格式"""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        # 如果已经是IP地址则直接返回
        if hostname and re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname):
            return url
            
        # 解析域名
        if hostname:
            ip = socket.gethostbyname(hostname)
            # 保留端口信息
            new_netloc = ip
            if parsed.port:
                new_netloc += f':{parsed.port}'
            # 重新构建URL
            parsed = parsed._replace(netloc=new_netloc)
            return urlunparse(parsed)
        return url
    except:
        return url  # 解析失败返回原始URL

# 定义LSTM模型类（必须与训练时的定义完全一致）
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                                batch_first=True, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(hidden)

# 配置参数（需要与训练时相同）
max_length = 100  # 必须与训练时设置的max_length一致

# 加载模型和字符映射
model = load('Secure_Cyber_Systems/lstm_train/phishing_detector_lstm.joblib')
char_to_idx = load('Secure_Cyber_Systems/lstm_train/char_to_idx.joblib')

# 确保模型在CPU并设置为评估模式
model = model.to('cpu')
model.eval()

def preprocess_url(url, char_to_idx, max_length):
    """将URL转换为模型需要的输入张量"""
    # 域名解析处理
    processed_url = resolve_domain(url)
    
    # 转换为字符索引序列
    sequence = [char_to_idx.get(char, 0) for char in processed_url]  # 0表示未知字符
    
    # 填充或截断序列
    if len(sequence) < max_length:
        padded = sequence + [0] * (max_length - len(sequence))
    else:
        padded = sequence[:max_length]
    
    # 转换为PyTorch张量并添加batch维度
    return torch.LongTensor([padded])

def predict_url(url):
    """预测单个URL的分类"""
    # 预处理
    input_tensor = preprocess_url(url, char_to_idx, max_length)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor.to('cpu'))
        probability = torch.sigmoid(output).item()
    
    # 解析结果（假设1表示钓鱼网站，0表示正常）
    return {
        'original_url': url,
        'processed_url': resolve_domain(url),  # 显示处理后的URL
        'phishing_probability': probability,
        'prediction': 'phishing' if probability > 0.5 else 'legitimate'
    }

if __name__ == "__main__":
    # 示例预测
    test_urls = [
        "http://example.com",
        "http://phishing-site.com/login.php?user=admin&password=1234",
        "https://www.trusted-bank.com/online-banking/",
        "https://128.199.211.187/robots.txt",
        "https://www.zhihu.com/?lang=zh-Hant"
    ]

    for url in test_urls:
        result = predict_url(url)
        print(f"Original URL: {result['original_url']}")
        print(f"Processed URL: {result['processed_url']}")
        print(f"Phishing Probability: {result['phishing_probability']:.4f}")
        print(f"Prediction: {result['prediction']}\n{'-'*50}")