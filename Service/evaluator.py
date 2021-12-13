# coding=utf-8
"""
@author: Oscar
@license: (C) Copyright 2019-2022, ZJU.
@contact: 499616042@qq.com
@software: pycharm
@file: evaluator.py
@time: 2020/9/2 15:19
"""
import torch
import logging
import numpy as np
from tqdm import tqdm
logger = logging.getLogger(__name__)


def get_base_out(model, loader, device, task_type):
    """
    每一个任务的 forward 都一样，封装起来
    """
    model.eval()

    with torch.no_grad():
        for idx, _batch in enumerate(tqdm(loader, desc=f'Get {task_type} task predict logits')):

            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)

            tmp_out = model(**_batch)

            yield tmp_out


def pointer_decode(logits, raw_text, start_threshold=0.5, end_threshold=0.5):
    candidate_entities = []

    start_ids = np.argwhere(logits[:, 0] > start_threshold)[:, 0]
    end_ids = np.argwhere(logits[:, 1] > end_threshold)[:, 0]
    raw_text_tokens = raw_text.split(" ")
    # 选最短的
    for _start in start_ids:
        for _end in end_ids:
            # 限定 trigger 长度不能超过 3
            if _end >= _start and _end - _start <= 2:
                # (start, end, start_logits + end_logits)
                candidate_entities.append(
                    (" ".join(raw_text_tokens[_start: _end + 1]), _start, logits[_start][0] + logits[_end][1]))
                break

    entities = []

    if len(candidate_entities):
        candidate_entities = sorted(candidate_entities, key=lambda x: x[-1], reverse=True)
        for _ent in candidate_entities:
            entities.append(_ent[:2])
    else:
        # 最后还是没有解码出 trigger 时选取 logits 最大的作为 trigger
        start_ids = np.argmax(logits[:, 0])
        end_ids = np.argmax(logits[:, 1])

        if end_ids < start_ids:
            end_ids = start_ids + np.argmax(logits[start_ids:, 1])

        entities.append((" ".join(raw_text_tokens[start_ids: end_ids + 1]), int(start_ids)))

    return entities


def pointer_crf_decode(logits, raw_text, start_threshold=0.5, end_threshold=0.5, force_decode=False):
    """
    :param logits:          sub / obj 最后输出的 logits，第一行为 start 第二行为 end
    :param raw_text:        原始文本
    :param start_threshold: logits start 位置大于阈值即可解码
    :param end_threshold:   logits end 位置大于阈值即可解码
    :param force_decode:    强制解码输出
    :return:
    [(entity, offset),...]
    """
    entities = []
    candidate_entities = []

    start_ids = np.argwhere(logits[:, 0] > start_threshold)[:, 0]
    end_ids = np.argwhere(logits[:, 1] > end_threshold)[:, 0]
    raw_text_token = raw_text.split(" ")
    # 选最短的
    for _start in start_ids:
        for _end in end_ids:
            if _end >= _start:
                # (start, end, logits)
                candidate_entities.append((_start, _end, logits[_start][0] + logits[_end][1]))
                break
    candidate_entities = sorted(candidate_entities, key=lambda x: x[-1], reverse=True)
    # 找整个候选集，如果存在包含的实体对选 logits 最高的作为候选
    for x in candidate_entities:
        flag = True
        for y in candidate_entities:
            if x == y:
                continue

            text_x = " ".join(raw_text_token[x[0]:x[1] + 1])
            text_y = " ".join(raw_text_token[y[0]:y[1] + 1])

            if text_x in text_y or text_y in text_x:
                if y[2] > x[2]:
                    flag = False
                    break
        if flag:
            entities.append((" ".join(raw_text_token[x[0]:x[1] + 1]), int(x[0])))

    if force_decode and not len(entities):
        start_ids = np.argmax(logits[:, 0])
        end_ids = np.argmax(logits[:, 1])

        if end_ids < start_ids:
            end_ids = start_ids + np.argmax(logits[start_ids:, 1])

        entities.append((" ".join(raw_text_token[start_ids: end_ids + 1]), int(start_ids)))

    return entities


def calculate_metric(gt, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gt:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1

    fn = len(gt) - tp

    return np.array([tp, fp, fn])


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])


def evaluation(model, dev_info, device, **kwargs):
    """
    线下评估 trigger 模型
    """
    dev_loader, dev_callback_info = dev_info

    pred_logits = None

    for tmp_pred in get_base_out(model, dev_loader, device):
        tmp_pred = tmp_pred[0].cpu().numpy()

        if pred_logits is None:
            pred_logits = tmp_pred
        else:
            pred_logits = np.append(pred_logits, tmp_pred, axis=0)

    assert len(pred_logits) == len(dev_callback_info)

    start_threshold = kwargs.pop('start_threshold')
    end_threshold = kwargs.pop('end_threshold')

    zero_pred = 0

    tp, fp, fn = 0, 0, 0

    for tmp_pred, tmp_callback in zip(pred_logits, dev_callback_info):
        text, gt_triggers, distant_triggers = tmp_callback
        tmp_pred = tmp_pred[1:1 + len(text.split(" "))]

        pred_triggers = pointer_decode(tmp_pred, text,
                                       start_threshold=start_threshold,
                                       end_threshold=end_threshold)

        if not len(pred_triggers):
            zero_pred += 1

        tmp_tp, tmp_fp, tmp_fn = calculate_metric(gt_triggers, pred_triggers)

        tp += tmp_tp
        fp += tmp_fp
        fn += tmp_fn

    p, r, f1 = get_p_r_f(tp, fp, fn)

    metric_str = f'In start threshold: {start_threshold}; end threshold: {end_threshold}\n'
    metric_str += f'[MIRCO] precision: {p:.4f}, recall: {r:.4f}, f1: {f1:.4f}\n'
    metric_str += f'Zero pred nums: {zero_pred}'

    return metric_str, f1


