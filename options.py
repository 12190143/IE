# coding=utf-8
import argparse


class BaseArgs:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        # args for path
        parser.add_argument('--raw_data_dir', default='',
                            help='the data dir of raw data')

        parser.add_argument('--mid_data_dir', default='',
                            help='the mid data dir')

        parser.add_argument('--output_dir', default='./out/',
                            help='the output dir for model checkpoints')

        parser.add_argument('--bert_dir', default='../bert/torch_roberta_wwm',
                            help='bert dir for ernie / roberta-wwm / uer / semi-bert')

        parser.add_argument('--bert_type', default='roberta_wwm',
                            help='roberta_wwm / ernie_1 / uer_large for bert')

        parser.add_argument('--dataset', default='opener_en',
                            help='')

        # other args
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids to use, -1 for cpu, "1, 3" for multi gpu')

        parser.add_argument('--mode', type=str, default='train',
                            help='train / test / stack (train / dev)')

        parser.add_argument('--task_type', type=str, default='trigger',
                            help='trigger / role1 & role2 / attribution task for event extraction')

        # args used for train / dev

        parser.add_argument('--max_seq_len', default=256, type=int)

        parser.add_argument('--eval_batch_size', default=64, type=int)

        parser.add_argument('--start_threshold', default=0.5, type=float,
                            help='threshold of entity start when decoding')

        parser.add_argument('--end_threshold', default=0.5, type=float,
                            help='threshold of entity end when decoding')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()


class TrainArgs(BaseArgs):
    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser = BaseArgs.initialize(parser)

        parser.add_argument('--train_epochs', default=10, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')

        parser.add_argument('--lr', default=2e-5, type=float,
                            help='learning rate for the bert module')

        parser.add_argument('--other_lr', default=2e-4, type=float,
                            help='learning rate for the module except bert')

        parser.add_argument('--max_grad_norm', default=1.0, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0., type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=64, type=int)

        parser.add_argument('--eval_model', default=False, action='store_true',
                            help='whether to eval model after training')

        return parser


class DevArgs(BaseArgs):
    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser = BaseArgs.initialize(parser)

        parser.add_argument('--dev_dir', type=str, help='dev model dir')

        # used for preliminary data forward
        parser.add_argument('--dev_dir_trigger', type=str, help='dev model dir')
        parser.add_argument('--dev_dir_role', type=str, help='dev model dir')

        return parser


class TestArgs(BaseArgs):
    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser = BaseArgs.initialize(parser)

        parser.add_argument('--version', default='v0', type=str,
                            help='submit version')

        parser.add_argument('--submit_dir', default='./submit', type=str)

        parser.add_argument('--ckpt_dir', required=True, type=str)

        parser.add_argument('--start_threshold', default=0.5, type=float)

        parser.add_argument('--end_threshold', default=0.5, type=float)

        return parser
