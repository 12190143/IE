from flask import Flask, jsonify, request
import sys

sys.path.append("/workspace/IE")
import time
import json
import numpy as np
import torch
from Service.model_utils import SeviceModel
from transformers import AdamW, get_linear_schedule_with_warmup
from functions_utils import load_model_and_parallel, save_model
from Service.trainer import train_batch as service_train_batch
from Service.trainer import forward_batch as service_forward_batch
from options import TrainArgs
from functions_utils import tensor_to_list, list_to_tensor, array_to_tensor, tensor_to_array
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


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


def service_api(json_data):
    # json_data = request.json
    userID = json_data['userID']
    batch_data = json_data['batch_data']
    batch_data = array_to_tensor(batch_data, device=device)
    # for key in batch_data.keys():
    #     batch_data[key] = batch_data[key].to(device)
    batch_data['inputs_embeds'].requires_grad = True
    gradient, loss = service_train_batch(opt, service_model, service_optimizer, service_scheduler, batch_data,
                                         use_n_gpus=False)
    batch_data['gradient'] = gradient
    batch_data.pop("output", None)
    batch_data = tensor_to_array(batch_data, data_type=np.float32)
    # print("service loss: {}".format(loss), flush=True)
    return {
        "msg": "success",
        'userID': userID,
        "batch_data": batch_data
    }


def service_forward(json_data):
    # json_data = request.json
    userID = json_data['userID']
    batch_data = json_data['batch_data']
    batch_data = array_to_tensor(batch_data, device=device)
    # for key in batch_data.keys():
    #     batch_data[key] = batch_data[key].to(device)
    output = service_forward_batch(service_model, batch_data)
    batch_data['output'] = output
    batch_data = tensor_to_array(batch_data, data_type=np.float32)
    return {
        "msg": "success",
        'userID': userID,
        "batch_data": batch_data
    }


def service_save(json_data):
    # json_data = request.json
    # userID = json_data['userID']
    save_model(opt, service_model, type_name='service')
    return {
        "msg": "success",
        # 'userID': userID
    }
