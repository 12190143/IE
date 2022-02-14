# coding=utf-8
import os
import shutil
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import copy
import torch
import sys

sys.path.append("/workspace/IE")
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from processor import *
from plan_2.trainer import train_best
from options import TrainArgs, TestArgs
from Client.model_utils import ClientModel
from Service.model_utils import SeviceModel
from dataset_utils import MyDataset
from functions_utils import set_seed, get_model_path_list, load_model_and_parallel, \
    prepare_info, prepare_para_dict, array_to_tensor, tensor_to_array
import json
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
import pickle
from io import BytesIO
import gzip
import time
import requests
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


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


def read_test_data(file_path):
    examples = []
    with open(file_path, encoding='utf-8') as f:
        example = {
            "text": "",
            "labels": []
        }
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                if len(example['text']) > 0:
                    examples.append(example)
                    example = {
                        "text": "",
                        "labels": []
                    }
            else:
                word = line.split(" ")[0]
                label = line.split(" ")[1]
                example['text'] += word
                example['labels'].append(label)
    return examples


def fine_grade_tokenize(raw_text, tokenizer):
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)
    return tokens


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


def cal_size_of_json(json_data):
    ans = len(pickle.dumps(json_data)) / 1024.0
    size_json_format = {}
    batch_data = json_data['batch_data']
    for k in batch_data:
        val_size = len(pickle.dumps(batch_data[k]))
        size_json_format[k] = val_size / 1024.0
    size_json_format['all'] = ans
    return size_json_format


def pointer_decode(logits, text, type2id):
    id2type = {type2id[type_]: type_ for type_ in type2id.keys()}
    ans_tmp = np.argmax(logits, -1)
    BIO_Label = [id2type[i] for i in ans_tmp]
    entity_text = ""
    entity_type = None
    entity_start = 0
    entities = []
    for i in range(len(BIO_Label)):
        if BIO_Label[i].startswith("B"):
            entity_start = i
            entity_text = text[i]
            entity_type = BIO_Label[i].split("-")[1]
        elif BIO_Label[i].startswith("M"):
            entity_text += text[i]
        elif BIO_Label[i].startswith("E"):
            entity_text += text[i]
            if len(entity_text):
                entities.append([entity_text, entity_type, entity_start])
            entity_text = ""
    return entities


def pipeline_predict(opt):
    """
    pipeline predict the submit results
    """
    if not os.path.exists(opt.submit_dir):
        os.makedirs(opt.submit_dir)

    submit = []

    text_examples = read_test_data(os.path.join(opt.raw_data_dir, '{}/test.char.bmes'.format(opt.dataset)))[: 20]

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=opt.bert_dir, add_prefix_space=True)

    with open("type2id.json", "r", encoding='utf-8') as f:
        type2id = json.load(f)

    # id2type = {type2id[key]: key for key in type2id.keys()}

    headers = {
        'content-encoding': 'gzip',
        'accept-encoding': 'gzip',
        'Content-type': 'application/json;charset=UTF-8', 'Accept': 'text/plain'}
    client_forward_time_batch = None
    service_forward_time_batch = None
    hidden_size_batch = None
    submit = []
    for _ex in tqdm(text_examples, desc='decode test examples'):
        tmp_text = _ex['text']
        tmp_text_tokens = fine_grade_tokenize(tmp_text, tokenizer)

        encode_dict = tokenizer.encode_plus(text=tmp_text_tokens,
                                            max_length=50,
                                            pad_to_max_length=True,
                                            is_split_into_words=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        tmp_base_inputs = {'input_ids': torch.tensor([encode_dict['input_ids']]).long(),
                           'attention_mask': torch.tensor([encode_dict['attention_mask']]).long(),
                           'token_type_ids': torch.tensor([encode_dict['token_type_ids']]).long()}
        batch_data = copy.deepcopy(tmp_base_inputs)
        json_data = {"userID": 1, "batch_data": tensor_to_array(batch_data)}

        # -----------------------------------------------------
        t1 = time.time()
        buf = zip_pickle_compress(json_data)
        response = requests.post(url='http://localhost:5002/client_forward', data=buf.getvalue(),
                                 headers=headers)
        content_data = response.content
        content_length = response.request.headers['Content-Length']
        length = len(content_data)
        print("after client forward after zip ", length / 1024.0, float(content_length) / 1023.0, flush=True)
        json_data, bytes_data = unzip_pickle_compress(content_data)
        print("after client forward before zip ", len(bytes_data) / 1024.0)
        t2 = time.time()
        time_client_forward_intra = json_data["time"]
        client_forward_json_time = cal_time(t1, t2, time_client_forward_intra, type_name='client_forward')
        hidden_size = cal_size_of_json(json_data)
        client_forward_time_batch = add_json(client_forward_time_batch, client_forward_json_time)
        hidden_size_batch = add_json(hidden_size_batch, hidden_size)

        t1 = time.time()
        content_data = requests.post(url='http://localhost:5003/service_forward',
                                     data=zip_pickle_compress(json_data).getvalue(), headers=headers).content
        content_length = response.request.headers['Content-Length']
        length = len(content_data)
        print("after service forward after zip ", length / 1024.0, float(content_length) / 1023.0, flush=True)
        json_data, bytes_data = unzip_pickle_compress(content_data)
        print("after client forward before zip ", len(bytes_data) / 1024.0)
        t2 = time.time()
        time_service_forward_intra = json_data["time"]
        service_forward_json_time = cal_time(t1, t2, time_service_forward_intra, type_name='client_forward')
        service_forward_time_batch = add_json(service_forward_time_batch, service_forward_json_time)
        tmp_pred = json_data['batch_data']["output"]
        tmp_pred = tmp_pred[1:1 + len(tmp_text)]
        pred_triggers = pointer_decode(tmp_pred, tmp_text, type2id)
        submit.append({
            "text": tmp_text,
            "entity": pred_triggers
        })
    for k in client_forward_time_batch.keys():
        client_forward_time_batch[k] /= len(text_examples)
    for k in service_forward_time_batch.keys():
        service_forward_time_batch[k] /= len(text_examples)
    for k in hidden_size_batch.keys():
        hidden_size_batch[k] /= len(text_examples)
    print("hidden size: {}; \n".format(print_json(hidden_size_batch)), flush=True)
    print("client forward time: {};\n".format(print_json(client_forward_time_batch)),
          "service forward time: {}".format(print_json(service_forward_time_batch)), flush=True)
    if not os.path.exists(opt.submit_dir):
        os.makedirs(opt.submit_dir)

    with open(os.path.join(opt.submit_dir, f'{opt.dataset}_submit_{opt.version}.json'), 'w', encoding='utf-8') as f:
        json.dump(submit, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    args = TestArgs().get_parser()
    pipeline_predict(args)
