from flask import Flask, jsonify, request
import sys
from flask_compress import Compress
from io import BytesIO
import gzip
sys.path.append("/workspace/IE")
app = Flask(__name__)
# Compress(app)
import time
import json
import pickle


def zip_compress(data):
    buf = BytesIO()
    with gzip.GzipFile(mode='wb', fileobj=buf) as fp:
        gzip_value = json.dumps(data).encode()
        fp.write(gzip_value)
    return buf


def unzip_compress(data):
    buf = BytesIO(data)
    gf = gzip.GzipFile(fileobj=buf)
    content = gf.read().decode('UTF-8')
    return content


def zip_pickle_compress(data):
    buf = BytesIO()
    with gzip.GzipFile(mode='wb', fileobj=buf) as fp:
        gzip_value = pickle.dumps(data)
        fp.write(gzip_value)
    return buf


def unzip_pickle_compress(data):
    buf = BytesIO(data)
    gf = gzip.GzipFile(fileobj=buf)
    content = pickle.loads(gf.read())
    return content


@app.route('/service_save', methods=['POST'])
def service_save():
    print("****"*10)
    length = request.headers.get('Content-Length')
    # print(length)
    print("service_update_api_length: ", float(len(request.data)) / 1024.0, flush=True)
    content_encoding = request.headers['content-encoding']
    if content_encoding == 'gzip':
        json_data = unzip_pickle_compress(request.data)
    else:
        json_data = request.json

    ans = {
        "msg": "success",
        "length": float(length)/1024.0,
        "data_length": float(len(request.data)) / 1024.0,
        "batch_data": json_data
    }
    return zip_pickle_compress(ans).getvalue()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)