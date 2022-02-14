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
from functions_utils import tensor_to_list, list_to_tensor, tensor_to_array
from plan_2.evaluator_function import evaluation
from sys import getsizeof
from plan_2 import ClientFunction
from plan_2.ClientFunction import client_save, client_api, client_forward
from plan_2 import ServiceFunction
from plan_2.ServiceFunction import service_api, service_forward, service_save
import time

logger = logging.getLogger(__name__)


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
    for epoch in range(opt.train_epochs):

        batch_num = 0
        for step, batch_data in enumerate(train_loader):
            json_data = {"userID": 1, "batch_data": tensor_to_array(batch_data)}
            json_data = client_forward(json_data)
            json_data = service_api(json_data)
            json_data = client_api(json_data)
            batch_num += 1
            # json_data = eval(json_data)
            global_step += 1

        tmp_metric_str, tmp_f1, tmp_f1_bio = evaluation(dev_info)

        logger.info(f'In epoch {epoch}: {tmp_metric_str}')
        metric_str += f'In epoch {epoch}: {tmp_metric_str}' + '\n\n'

        if tmp_f1_bio > max_f1:
            max_f1 = tmp_f1_bio
            max_f1_step = epoch
            json_data = {"userID": 1}
            client_save(json_data)
            service_save(json_data)

    max_metric_str = f'Max f1 is: {max_f1}, in epoch {max_f1_step}'
    logger.info(max_metric_str)
    metric_str += max_metric_str + '\n'
    eval_save_path = os.path.join(opt.output_dir, 'eval_metric.txt')
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    with open(eval_save_path, 'a', encoding='utf-8') as f1:
        f1.write(metric_str)
    logger.info('Train done')
