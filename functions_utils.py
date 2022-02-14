# coding=utf-8
import os
import copy
import json
import torch
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_info(task_type, mid_data_dir):
    """
    prepare a dict for training in different task
    """

    info_dict = {}

    return info_dict


def prepare_para_dict(opt, info_dict):
    feature_para, dataset_para, model_para = {}, {}, {}
    task_type = opt.task_type

    if hasattr(opt, 'dropout_prob'):
        model_para['dropout_prob'] = opt.dropout_prob

    return feature_para, dataset_para, model_para


def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    """

    gpu_ids = gpu_ids.split(',')

    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    model.to(device)

    if len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')

    return model, device


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []

    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if 'model.pt' == _file:
                model_lists.append(os.path.join(root, _file))

    model_lists = sorted(model_lists,
                         key=lambda x: (x.split('/')[-3], int(x.split('/')[-2].split('-')[-1])))

    return model_lists


def tensor_to_list(data):
    for key in data.keys():
        data[key] = data[key].cpu().detach().numpy().tolist()
    return data


def tensor_to_array(data, data_type=np.float32):
    for key in data.keys():
        if key == "inputs_embeds" or key == "gradient":
            data[key] = np.array(data[key].cpu().detach().numpy(), dtype=data_type)
        else:
            data[key] = data[key].cpu().detach().numpy()
    return data


def list_to_tensor(data, device=None):
    if device:
        for key in data.keys():
            if key == "inputs_embeds" or key == "gradient":
                data[key] = torch.tensor(np.array(data[key])).float().to(device)
            else:
                data[key] = torch.tensor(np.array(data[key])).to(device)
            # print(data[key].dtype)
    else:
        for key in data.keys():
            if key == "inputs_embeds" or key == "gradient":
                data[key] = torch.tensor(np.array(data[key])).float()
            else:
                data[key] = torch.tensor(np.array(data[key]))
    return data


def array_to_tensor(data, device=None):
    if device:
        for key in data.keys():
            if key == "inputs_embeds" or key == "gradient":
                data[key] = torch.tensor(data[key]).float().to(device)
            else:
                data[key] = torch.tensor(data[key]).to(device)
            # print(data[key].dtype)
    else:
        for key in data.keys():
            if key == "inputs_embeds" or key == "gradient":
                data[key] = torch.tensor(data[key]).float()
            else:
                data[key] = torch.tensor(data[key])
    return data


def save_model(opt, model, global_step=None, type_name='client'):
    if global_step is None:
        output_dir = os.path.join(opt.output_dir, 'checkpoint-best')
    else:
        output_dir = os.path.join(opt.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        pass
        # shutil.rmtree(output_dir)
        # # os.rmdir(os.path.join(output_dir, 'model.pt'))
        # os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    print(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, '{}_model.pt'.format(type_name)))
