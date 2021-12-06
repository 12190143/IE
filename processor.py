# coding=utf-8
import copy
import json
import random
import logging
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

__all__ = ['Processor', 'ROLE2_TO_ID', 'fine_grade_tokenize', 'convert_examples_to_features']

ROLE2_TO_ID = {
    "O": 0,
    "B-time": 1,
    "I-time": 2,
    "E-time": 3,
    "S-time": 4,
    "B-loc": 5,
    "I-loc": 6,
    "E-loc": 7,
    "S-loc": 8,
    "X": 9
}


class BaseExample:
    def __init__(self,
                 set_type,
                 text,
                 label=None):
        self.set_type = set_type
        self.text = text
        self.label = label


class Example(BaseExample):
    def __init__(self,
                 set_type,
                 text,
                 label=None):
        super(Example, self).__init__(set_type=set_type, text=text, label=label)
        # self.trigger = trigger


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels


class Feature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 other_feature,
                 labels=None):
        super(Feature, self).__init__(token_ids=token_ids,
                                      attention_masks=attention_masks,
                                      token_type_ids=token_type_ids,
                                      labels=labels)
        self.other_feature = other_feature


class BaseProcessor:
    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            examples = json.load(f)
        return examples


class Processor(BaseProcessor):
    @staticmethod
    def _example_generator(raw_examples, set_type, max_seq_len):
        examples = []
        callback_info = []

        for _ex in raw_examples:
            text = _ex['text']
            labels = []
            for _ner in _ex['entities']:
                tmp_entity_text = _ner['text']
                tmp_entity_start = _ner['offset']
                if tmp_entity_start >= len(text.split(" ")) or tmp_entity_start >= max_seq_len:
                    continue
                labels.append([tmp_entity_text, tmp_entity_start])
                examples.append(Example(set_type=set_type,
                                        text=text,
                                        label=labels))
                callback_info.append((text, labels))

        if set_type == 'dev':
            return examples, callback_info
        else:
            return examples

    def get_train_examples(self, raw_examples, max_seq_len):
        return self._example_generator(raw_examples, 'train', max_seq_len)

    def get_dev_examples(self, raw_examples, max_seq_len):
        return self._example_generator(raw_examples, 'dev', max_seq_len)


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
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


def convert_example(ex_idx, example: Example, max_seq_len, tokenizer: BertTokenizer):
    """
    convert attribution example to attribution feature
    """
    set_type = example.set_type
    raw_text = example.text
    raw_label = example.label

    tokens = tokenizer.tokenize(raw_text)
    labels = [[0] * 2 for i in range(len(tokens))]  # start / end

    # tag labels
    for _label in raw_label:
        tmp_start = _label[1]
        tmp_end = _label[1] + len(_label[0].split(" ")) - 1

        labels[tmp_start][0] = 1
        labels[tmp_end][1] = 1

    if len(labels) > max_seq_len - 2:
        labels = labels[:max_seq_len - 2]

    pad_labels = [[0] * 2]
    labels = pad_labels + labels + pad_labels

    if len(labels) < max_seq_len:
        pad_length = max_seq_len - len(labels)
        labels = labels + pad_labels * pad_length

    # print(labels)
    assert len(labels) == max_seq_len

    encode_dict = tokenizer.encode_plus(text=raw_text,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        # is_split_into_words=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    # print(tokens)
    # print(token_ids)
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3 and set_type == 'train':
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text: {" ".join(tokens)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")
        logger.info(f'distant trigger: {distant_trigger_label}')

    feature = Feature(token_ids=token_ids,
                      attention_masks=attention_masks,
                      token_type_ids=token_type_ids,
                      labels=labels)

    return feature


def convert_examples_to_features(task_type, examples, bert_dir, max_seq_len, **kwargs):
    assert task_type in ['trigger', 'role1', 'attribution']

    # tokenizer = BertTokenizer.from_pretrained(bert_dir)
    try:
        tokenizer = BertTokenizer.from_pretrained(bert_dir)
    except:
        tokenizer = AutoTokenizer.from_pretrained(bert_dir, add_prefix_space=True)
        # tokenizer = PreTrainedTokenizerFast(tokenizer_file=bert_dir+"/tokenizer.json")
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.encode_plus()
        # tokenizer.add_special_tokens()
    logger.info(f'Vocab nums in this tokenizer is: {tokenizer.vocab_size}')

    features = []

    for i, example in enumerate(tqdm(examples, desc=f'convert examples')):
        if task_type == 'trigger':

            feature = convert_trigger_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer,
            )

        elif task_type == 'role1':
            feature = convert_role1_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer
            )

        else:
            feature = convert_attribution_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer,
                polarity2id=kwargs.get('polarity2id'),
                tense2id=kwargs.get('tense2id')
            )

        if feature is None:
            continue

        features.append(feature)

    return features
