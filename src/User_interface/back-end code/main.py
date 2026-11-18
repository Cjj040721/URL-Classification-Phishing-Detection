import io
from concurrent.futures.thread import ThreadPoolExecutor

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
from feature_extraction_for_new_URLs import getFeature
from lstm_v2_demo import predict_from_sample_row

app = Flask(__name__)
app.config['PREFERRED_URL_SCHEME'] = 'https'
CORS(app, resources={
    r"/detect": {"origins": "*"},
    r"/upload_csv": {"origins": "*"},
    r"/export_csv": {"origins": "*"}
})


@app.route('/')
def home():
    """处理根路径请求"""
    return "Welcome to the Phishing Detection System!"


@app.route('/favicon.ico')
def favicon():
    """提供favicon"""
    return app.send_static_file('favicon.ico')


@app.route('/detect', methods=['POST'])
def detect_phishing():
    """接收前端请求并返回预测结果"""
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    return predict_from_sample_row(getFeature(url))


def getResult(url):
    return predict_from_sample_row(getFeature(url))


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """接收前端上传的 CSV 文件，并逐行读取内容"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File is not a CSV file'}), 400

    try:
        df = pd.read_csv(file)

        if 'url' not in df.columns:
            return jsonify({'error': 'CSV file must contain "url" column'}), 400

        urls = [url for url in df['url'] if pd.notna(url)]

        with ThreadPoolExecutor(max_workers=100) as executor:
            results = list(executor.map(getResult, urls))

        return jsonify({
            'message': 'File processed successfully',
            'results': results,
            'total': len(results)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@app.route('/export_csv', methods=['POST'])
def export_csv():
    """接收JSON数据并返回CSV文件"""
    data = request.json
    if not data or 'data' not in data:
        return jsonify({'error': 'Invalid data format'}), 400

    try:
        df = pd.DataFrame(data['data'])

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=export.csv"}
        )
    except Exception as e:
        return jsonify({'error': f'Error exporting file: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(port=5001)
