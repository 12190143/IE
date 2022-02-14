from flask import Flask, jsonify, request
import sys
from flask_compress import Compress
from io import BytesIO
import gzip
import torch
import requests
sys.path.append("/workspace/IE")
app = Flask(__name__)
# Compress(app)
import time
import numpy as np
import pickle
import json


def zip_compress(data):
    buf = BytesIO()
    with gzip.GzipFile(mode='wb', fileobj=buf) as fp:
        gzip_value = json.dumps(data).encode()
        fp.write(gzip_value)
    return buf


def zip_pickle_compress(data):
    buf = BytesIO()
    with gzip.GzipFile(mode='wb', fileobj=buf) as fp:
        gzip_value = pickle.dumps(data)
        print("*****"*10, len(gzip_value)/1024.0)
        fp.write(gzip_value)
    return buf


def unzip_pickle_compress(data):
    buf = BytesIO(data)
    gf = gzip.GzipFile(fileobj=buf)
    bytes_data = gf.read()
    print("-----" * 10, len(bytes_data))
    content = pickle.loads(bytes_data)
    return content


def unzip_compress(data):
    buf = BytesIO(data)
    gf = gzip.GzipFile(fileobj=buf)
    bytes_data = gf.read()
    print("-----"*10, len(bytes_data)/1024.0)
    content = bytes_data.decode('UTF-8')
    return content


if __name__ == '__main__':
    headers = {
        'content-encoding': 'gzip',
        'accept-encoding': 'gzip',
        'Content-type': 'application/json;charset=UTF-8',
        'Accept': 'text/plain'
    }

    json_data = {
        "userID": 1,
        "data": torch.tensor(np.array(np.random.random((128, 768)), dtype=np.float32))
    }

    data = zip_pickle_compress(json_data)
    j_data = unzip_pickle_compress(data.getvalue())
    # print(j_data)

    print(len(zip_pickle_compress(json_data).getvalue())/1024)
    response = requests.post(url='http://localhost:5000/service_save', data=data.getvalue(), headers=headers)
    print(len(response.content)/1024)
    print(int(response.headers.get('Content-Length'))/1024.0)
    # print(unzip_pickle_compress(response.content))