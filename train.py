# coding=utf-8
import os
import shutil
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import os
import shutil
import copy
import torch
import logging
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from processor import *
from trainer import train_best
from options import TrainArgs
from Client.model_utils import ClientModel
from Service.model_utils import SeviceModel
from dataset_utils import MyDataset
from evaluator import evaluation
from functions_utils import set_seed, get_model_path_list, load_model_and_parallel, \
    prepare_info, prepare_para_dict


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def train_with_best(opt, processor, info_dict, train_examples, dev_info=None):
    feature_para, dataset_para, model_para = prepare_para_dict(opt, info_dict)

    train_features = processor.convert_examples_to_features(opt.task_type, train_examples, opt.bert_dir,
                                                  opt.max_seq_len, **feature_para)

    logger.info(f'Build {len(train_features)} train features')

    train_dataset = MyDataset(train_features, 'train')

    client_model = ClientModel(opt.bert_dir, dropout_prob=opt.dropout_prob, k=2)
    service_model = SeviceModel(opt.bert_dir, dropout_prob=opt.dropout_prob, k=2)

    dev_examples, dev_callback_info = dev_info

    dev_features = processor.convert_examples_to_features(opt.task_type, dev_examples, opt.bert_dir,
                                                opt.max_seq_len, **feature_para)

    logger.info(f'Build {len(dev_features)} dev features')

    dev_dataset = MyDataset(dev_features, 'dev')

    dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,
                            shuffle=False, num_workers=8)

    dev_info = (dev_loader, dev_callback_info)

    train_best(opt, client_model, service_model, train_dataset, dev_info, info_dict)


def training(opt):
    # processor = Processor()
    processor = MSRANerProcessor()
    info_dict = prepare_info(opt.task_type, opt.mid_data_dir)

    # train_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, '{}/train.json'.format(opt.dataset)), set_type='train')
    train_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, '{}/train_dev.char.bmes'.format(opt.dataset)), set_type='train')
    train_examples = processor.get_train_examples(train_raw_examples, max_seq_len=opt.max_seq_len)

    dev_info = None
    if opt.eval_model:
        # dev_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, '{}/dev.json'.format(opt.dataset)))
        dev_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, '{}/test.char.bmes'.format(opt.dataset)))
        dev_info = processor.get_dev_examples(dev_raw_examples, max_seq_len=opt.max_seq_len)

    train_with_best(opt, processor, info_dict, train_examples, dev_info)


if __name__ == '__main__':
    args = TrainArgs().get_parser()

    training(args)
