# coding=utf-8
import os
import shutil
import copy
import sys
import json

sys.path.append("/workspace/IE")
import requests
import torch
import logging
from torch.utils.data import DataLoader, RandomSampler
from functions_utils import tensor_to_list, list_to_tensor, tensor_to_array, array_to_tensor
from plan_2.evaluator import evaluation
from sys import getsizeof
import time
import pickle
from io import BytesIO
import gzip
# from pympler import asizeof
logger = logging.getLogger(__name__)


def get_list_size(data):
    # val_size = sys.getsizeof(pickle.dumps(data))
    val_size = 0.0
    for a in data:
        if type(a[0]).__name__ == 'list':
            for b in a:
                if type(b[0]).__name__ == 'list':
                    for c in b:
                        val_size += sys.getsizeof(c)
                else:
                    val_size += sys.getsizeof(b)
        else:
            val_size += sys.getsizeof(a)
    return val_size


def cal_size_of_json(json_data):
    ans = len(pickle.dumps(json_data)) / 1024.0
    size_json_format = {}
    batch_data = json_data['batch_data']
    for k in batch_data:
        val_size = len(pickle.dumps(batch_data[k]))
        size_json_format[k] = val_size / 1024.0
    size_json_format['all'] = ans
    return size_json_format


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
        fp.write(gzip_value)
    return buf


def unzip_pickle_compress(data):
    buf = BytesIO(data)
    gf = gzip.GzipFile(fileobj=buf)
    bytes_data = gf.read()
    content = pickle.loads(bytes_data)
    return content, bytes_data


def unzip_compress(data):
    buf = BytesIO(data)
    gf = gzip.GzipFile(fileobj=buf)
    content = gf.read().decode('UTF-8')
    return content


def cal_time(t1, t2, time_intra, type_name="client_forward"):
    total_time = t2 - t1
    send_time = time_intra[0] - t1
    get_time = t2 - time_intra[-1]
    ans = {
        "all": total_time,
        'send_time': send_time,
        'get_time': get_time
    }
    json2tensor_time = time_intra[1] - time_intra[0]
    cal_time = time_intra[2] - time_intra[1]
    tensor2json_time = time_intra[3] - time_intra[2]
    ans['json2tensor_time'] = json2tensor_time
    ans['tensor2json_time'] = tensor2json_time
    if type_name == "client_forward":
        ans['client_forward_time'] = cal_time
    elif type_name == "client_update":
        ans['client_forward_backward_time'] = cal_time
    elif type_name == "service_update":
        ans['service_forward_backward_time'] = cal_time
    return ans


def print_json(json_data):
    tmp = []
    for k in json_data.keys():
        tmp.append("{}: {}".format(k, json_data[k]))
    ans = "; ".join(tmp)
    # print(ans, flush=True)
    return ans


def add_json(total_json, tmp_json):
    if total_json is None:
        total_json = tmp_json
    else:
        for k in total_json:
            total_json[k] += tmp_json[k]
    return total_json


def train_best(opt, train_dataset, dev_info, info_dict):
    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              sampler=train_sampler,
                              num_workers=8)

    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {opt.train_epochs}")
    logger.info(f"  Total training batch size = {opt.train_batch_size}")
    t_total = len(train_loader) * opt.train_epochs
    opt.t_total = t_total
    global_step = 0

    max_f1 = 0.
    max_f1_step = 0
    metric_str = ''
    headers = {
        'content-encoding': 'gzip',
        'accept-encoding': 'gzip',
        'Content-type': 'application/json;charset=UTF-8', 'Accept': 'text/plain'}
    for epoch in range(opt.train_epochs):
        client_forward_time_batch = None
        service_update_time_batch = None
        client_update_time_batch = None
        hidden_size_batch = None
        gradient_size_batch = None
        size_batch = [0.0, 0.0]
        batch_num = 0
        for step, batch_data in enumerate(train_loader):
            json_data = {"userID": 1, "batch_data": tensor_to_array(batch_data)}
            t1 = time.time()
            # -----------------------------------------------------
            buf = zip_pickle_compress(json_data)
            response = requests.post(url='http://localhost:5001/client_forward', data=buf.getvalue(),
                                      headers=headers)
            content_data = response.content
            content_length = response.request.headers['Content-Length']
            length = len(content_data)
            print("after client forward after zip ", length/1024.0, float(content_length)/1023.0, flush=True)
            json_data, bytes_data = unzip_pickle_compress(content_data)
            print("after client forward before zip ", len(bytes_data)/1024.0)
            t2 = time.time()
            time_client_forward_intra = json_data["time"]
            client_forward_json_time = cal_time(t1, t2, time_client_forward_intra, type_name='client_forward')
            hidden_size = cal_size_of_json(json_data)

            client_forward_time_batch=add_json(client_forward_time_batch, client_forward_json_time)
            hidden_size_batch = add_json(hidden_size_batch, hidden_size)

            # -----------------------------------------------------
            buf = zip_pickle_compress(json_data)
            t1 = time.time()
            response = requests.post(url='http://localhost:5000/service_update', data=buf.getvalue(),
                                      headers=headers)
            content_data = response.content
            content_length = response.request.headers['Content-Length']
            length = len(content_data)
            print("after service update after zip ", length/1024.0, float(content_length)/1024.0, flush=True)
            json_data, bytes_data = unzip_pickle_compress(content_data)
            print("after service update before zip ", len(bytes_data)/1024.0)
            t2 = time.time()
            time_service_update_intra = json_data["time"]
            service_update_json_time = cal_time(t1, t2, time_service_update_intra, type_name='service_update')
            gradient_size = cal_size_of_json(json_data)

            service_update_time_batch = add_json(service_update_time_batch, service_update_json_time)
            gradient_size_batch = add_json(gradient_size_batch, gradient_size)

            # -----------------------------------------------------
            buf = zip_pickle_compress(json_data)
            print("hidden size: {};\n".format(print_json(hidden_size)), "gradient size: {}".format(print_json(gradient_size)), flush=True)
            t1 = time.time()
            response = requests.post(url='http://localhost:5001/client_update', data=buf.getvalue(),
                                      headers=headers)
            content_data = response.content
            content_length = response.request.headers['Content-Length']
            length = len(content_data)
            json_data, _ = unzip_pickle_compress(content_data)
            t2 = time.time()
            time_client_update_intra = json_data["time"]
            client_update_json_time = cal_time(t1, t2, time_client_update_intra, type_name='client_update')
            client_update_time_batch = add_json(client_update_time_batch, client_update_json_time)

            print("client forward: {};\n".format(print_json(client_forward_json_time)),
                  "service update: {};\n".format(service_update_json_time),
                  "client update: {}\n".format(client_update_json_time), flush=True)
            batch_num += 1
            # json_data = eval(json_data)
            global_step += 1

        for k in client_forward_time_batch.keys():
            client_forward_time_batch[k] /= batch_num
        for k in client_update_time_batch.keys():
            client_update_time_batch[k] /= batch_num
        for k in service_update_time_batch.keys():
            service_update_time_batch[k] /= batch_num
        for k in hidden_size_batch.keys():
            hidden_size_batch[k] /= batch_num
        for k in gradient_size_batch.keys():
            gradient_size_batch[k] /= batch_num
        print("hidden size: {}; \n".format(print_json(hidden_size_batch)), "gradient size: {}".format(print_json(gradient_size_batch)), flush=True)
        print("client forward time: {};\n".format(print_json(client_forward_time_batch)), "service update time: {}".format(print_json(service_update_time_batch)),
              "client update time: {};".format(print_json(client_update_time_batch)), flush=True)
        tmp_metric_str, tmp_f1, tmp_f1_bio = evaluation(dev_info)

        logger.info(f'In epoch {epoch}: {tmp_metric_str}')
        metric_str += f'In epoch {epoch}: {tmp_metric_str}' + '\n\n'

        if tmp_f1_bio > max_f1:
            max_f1 = tmp_f1_bio
            max_f1_step = epoch
            json_data = {"userID": 1}
            requests.post(url='http://localhost:5001/client_save', data=zip_pickle_compress(json_data).getvalue(), headers=headers)
            requests.post(url='http://localhost:5000/service_save', data=zip_pickle_compress(json_data).getvalue(), headers=headers)

    max_metric_str = f'Max f1 is: {max_f1}, in epoch {max_f1_step}'
    logger.info(max_metric_str)
    metric_str += max_metric_str + '\n'
    eval_save_path = os.path.join(opt.output_dir, 'eval_metric.txt')
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    with open(eval_save_path, 'a', encoding='utf-8') as f1:
        f1.write(metric_str)

    # clear cuda cache to avoid OOM
    # torch.cuda.empty_cache()

    logger.info('Train done')
