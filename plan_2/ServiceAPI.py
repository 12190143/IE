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
import numpy as np
import pickle
import torch
from Service.model_utils import SeviceModel
from transformers import AdamW, get_linear_schedule_with_warmup
from functions_utils import load_model_and_parallel, save_model
from Service.trainer import train_batch as service_train_batch
from Service.trainer import forward_batch as service_forward_batch
from options import TrainArgs
from functions_utils import tensor_to_list, list_to_tensor, tensor_to_array, array_to_tensor
import os


def build_optimizer_and_scheduler(opt, model):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'service_model':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * opt.t_total), num_training_steps=opt.t_total
    )

    return optimizer, scheduler


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


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


def build_model(opt):
    service_model = SeviceModel(opt.bert_dir, dropout_prob=opt.dropout_prob, k=2)
    service_model, device = load_model_and_parallel(service_model, opt.gpu_ids)
    use_n_gpus = False
    if hasattr(service_model, "module"):
        use_n_gpus = True
    service_optimizer, service_scheduler = build_optimizer_and_scheduler(opt, service_model)
    return service_model, service_optimizer, service_scheduler, device, use_n_gpus


opt = TrainArgs().get_parser()
service_model, service_optimizer, service_scheduler, device, use_n_gpus = build_model(opt=opt)
time.sleep(30)


@app.route('/service_update', methods=['POST'])
def service_api():
    t1 = time.time()
    length = request.headers.get('Content-Length')
    print("before service_update_api_length: ", float(len(request.data))/1024.0, float(length)/1024.0, flush=True)
    content_encoding = request.headers['content-encoding']
    if content_encoding == 'gzip':
        json_data = unzip_pickle_compress(request.data)
    else:
        json_data = request.json
    userID = json_data['userID']
    batch_data = json_data['batch_data']
    batch_data = array_to_tensor(batch_data, device=device)
    t2 = time.time()
    batch_data['inputs_embeds'].requires_grad = True
    gradient, loss = service_train_batch(opt, service_model, service_optimizer, service_scheduler, batch_data,
                                         use_n_gpus=False)
    t3 = time.time()
    batch_data['gradient'] = gradient
    batch_data.pop("output", None)
    batch_data = tensor_to_array(batch_data)
    t4 = time.time()
    print("before service loss: {}".format(loss), flush=True)
    ans = {
        "msg": "success",
        'userID': userID,
        "batch_data": batch_data,
        "time": [t1, t2, t3, t4]
    }
    return zip_pickle_compress(ans).getvalue()


@app.route('/service_forward', methods=['POST'])
def service_forward():
    t1 = time.time()
    length = request.headers.get('Content-Length')
    print("before servive_forward_api_length: ", float(len(request.data))/1024.0, float(length)/1024.0, flush=True)
    content_encoding = request.headers['content-encoding']
    if content_encoding == 'gzip':
        json_data = unzip_pickle_compress(request.data)
    else:
        json_data = request.json

    userID = json_data['userID']
    batch_data = json_data['batch_data']
    batch_data = array_to_tensor(batch_data, device=device)
    t2 = time.time()
    output = service_forward_batch(service_model, batch_data)
    t3 = time.time()
    batch_data['output'] = output
    batch_data = tensor_to_array(batch_data)
    t4 = time.time()
    ans = {
        "msg": "success",
        'userID': userID,
        "batch_data": batch_data,
        "time": [t1, t2, t3, t4]
    }
    return zip_pickle_compress(ans).getvalue()


@app.route('/service_save', methods=['POST'])
def service_save():
    t1 = time.time()
    # json_data = request.json
    # userID = json_data['userID']
    save_model(opt, service_model, type_name='service')
    t2 = time.time()
    ans = {
        "msg": "success",
        # 'userID': userID
        "time": [t1, t2]
    }
    return zip_pickle_compress(ans).getvalue()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
