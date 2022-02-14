import requests
from flask import Flask, jsonify, request
from flask_compress import Compress
import sys
from io import BytesIO
import gzip
sys.path.append("/workspace/IE")
app = Flask(__name__)
# Compress(app)

import time
import json
import numpy as np
import torch
import pickle
import os
from Client.model_utils import ClientModel
from transformers import AdamW, get_linear_schedule_with_warmup
from functions_utils import load_model_and_parallel, save_model
from Client.trainer import train_batch as client_train_batch
from Client.trainer import forward_batch as client_forward_batch
from options import TrainArgs
from functions_utils import tensor_to_list, list_to_tensor, array_to_tensor, tensor_to_array


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
        if space[0] == 'client_model':
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


def build_model(opt):
    client_model = ClientModel(opt.bert_dir, dropout_prob=opt.dropout_prob, k=2)
    client_model, device = load_model_and_parallel(client_model, opt.gpu_ids)
    use_n_gpus = False
    if hasattr(client_model, "module"):
        use_n_gpus = True
    client_optimizer, client_scheduler = build_optimizer_and_scheduler(opt, client_model)
    return client_model, client_optimizer, client_scheduler, device, use_n_gpus


opt = TrainArgs().get_parser()
client_model, client_optimizer, client_scheduler, device, use_n_gpus = build_model(opt=opt)
time.sleep(60)
output_dir = os.path.join(opt.output_dir, 'checkpoint-best')
client_model, _ = load_model_and_parallel(client_model, opt.gpu_ids[0],
                                                   ckpt_path=os.path.join(output_dir, 'client_model.pt'))
time.sleep(60)


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


@app.route('/client_forward', methods=['POST'])
def client_forward():
    t1 = time.time()
    length = request.headers.get('Content-Length')
    # print("**************", length)
    print("before client_forward_api_length: ", float(len(request.data))/1024.0, float(length)/1024, flush=True)
    content_encoding = request.headers['content-encoding']
    # print ("********", content_encoding)
    if content_encoding == 'gzip':
        json_data = unzip_pickle_compress(request.data)
    else:
        json_data = request.json

    userID = json_data['userID']
    batch_data = json_data['batch_data']
    batch_data = array_to_tensor(batch_data, device=device)
    t2 = time.time()
    client_output = client_forward_batch(client_model, batch_data)
    t3 = time.time()
    service_input = client_output.detach().clone()  # .requires_grad_(True)
    service_input.requires_grad = True
    batch_data['inputs_embeds'] = service_input
    batch_data = tensor_to_array(batch_data, data_type=np.float16)
    t4 = time.time()
    ans = {
        "msg": "success",
        'userID': userID,
        "batch_data": batch_data,
        "time": [t1, t2, t3, t4]
    }
    return zip_pickle_compress(ans).getvalue()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
