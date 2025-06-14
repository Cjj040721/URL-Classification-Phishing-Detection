import io
from concurrent.futures.thread import ThreadPoolExecutor

from flask import Flask, request, jsonify, Response
import re
from urllib.parse import urlparse, urlunparse
from joblib import load
import torch
import torch.nn as nn
from flask_cors import CORS
import socket
import pandas as pd
from feature_extraction_for_new_URLs import getFeature
from lstm_v2_demo import predict_from_sample_row
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)
app.config['PREFERRED_URL_SCHEME'] = 'https'
CORS(app, resources={
    r"/detect": {"origins": "*"},
    r"/upload_csv": {"origins": "*"},  # 新增上传文件的跨域支持
    r"/export_csv": {"origins": "*"}
})

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

# 加载训练好的模型
baseModel = load('phishing_detector_lstm(1).joblib')
baseModel = baseModel.to('cuda')
baseModel.eval()
char_to_idx = load('char_to_idx.joblib')
max_length = 100

@app.route('/')
def home():
    """
    处理根路径请求
    """
    return "Welcome to the Phishing Detection System!"

@app.route('/favicon.ico')
def favicon():
    """
    提供favicon
    """
    return app.send_static_file('favicon.ico')

@app.route('/detect', methods=['POST'])
def detect_phishing():
    """
    接收前端请求并返回预测结果
    """
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    #用的快model，要用准model换成下面的代码
    #result = predict_url(url)
    return predict_from_sample_row(getFeature(url))
    return result

def getResult(url):
    return predict_from_sample_row(getFeature(url))
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """
    接收前端上传的 CSV 文件，并逐行读取内容
    """
    # 检查是否上传了文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 确保上传的是 CSV 文件
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File is not a CSV'}), 400

    try:
        # 使用 pandas 读取 CSV 文件
        df = pd.read_csv(file)

        # 检查是否存在url列
        if 'url' not in df.columns:
            return jsonify({'error': 'CSV file must contain "url" column'}), 400

        # 对每个URL进行预测
        results = []
        #用的快model，要用准model换成下面的代码
        #for url in df['url']:
        #    if pd.notna(url):  # 跳过空值
        #        result = predict_url(url)
        #        results.append(result)

        for url in df['url']:
            if pd.notna(url):  # 跳过空值
                result = predict_from_sample_row(getFeature(url))
                results.append(result)

        urls = [url for url in df['url'] if pd.notna(url)]

        with ThreadPoolExecutor(max_workers=100) as executor:
            results = list(executor.map(getResult, urls))

        # 返回所有预测结果
        return jsonify({
            'message': 'File processed successfully',
            'results': results,
            'total': len(results)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


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
        output = baseModel(input_tensor.to('cuda'))
        probability = torch.sigmoid(output).item()

    # 解析结果（假设1表示钓鱼网站，0表示正常）
    return {
        'original_url': url,
        'processed_url': resolve_domain(url),  # 显示处理后的URL
        'phishing_probability': probability,
        'prediction': 'phishing' if probability > 0.5 else 'legitimate'
    }


# ... 前面的代码保持不变 ...

@app.route('/export_csv', methods=['POST'])
def export_csv():
    """
    接收JSON数据并返回CSV文件
    """
    data = request.json
    if not data or 'data' not in data:
        return jsonify({'error': 'Invalid data format'}), 400

    try:
        # 将数据转换为DataFrame
        df = pd.DataFrame(data['data'])

        # 创建内存中的CSV文件
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # 返回CSV文件
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=export.csv"}
        )
    except Exception as e:
        return jsonify({'error': f'Error exporting CSV: {str(e)}'}), 500


# ... 后面的代码保持不变 ...

if __name__ == '__main__':
    app.run()

