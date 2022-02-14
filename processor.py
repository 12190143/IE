# coding=utf-8
import copy
import json
import random
import logging
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['Processor', "MSRANerProcessor"]


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
                 labels=None):
        super(Feature, self).__init__(token_ids=token_ids,
                                      attention_masks=attention_masks,
                                      token_type_ids=token_type_ids,
                                      labels=labels)
        # self.other_feature = other_feature
        pass


class BaseProcessor:
    def __init__(self):
        self.type2id = {}

    # @staticmethod
    def read_json(self, file_path, set_type=None):
        examples = []
        with open(file_path, encoding='utf-8') as f:
            for line in f.readlines():
                examples.append(json.loads(line.strip()))

        if set_type == 'train':
            type2id = {"O": 0}
            class_index = 1
            for _ex in examples:
                for entity_type in _ex['label'].keys():
                    label = 'B-' + entity_type
                    if label not in type2id.keys():
                        type2id['B-' + entity_type] = class_index
                        class_index += 1
                    label = 'I-' + entity_type
                    if label not in type2id.keys():
                        type2id['I-' + entity_type] = class_index
                        class_index += 1
                    # if 'E' + entity_type not in type2id:
                    #     type2id['E' + entity_type] = len(type2id)
            self.type2id = type2id
            with open("type2id.json", 'w', encoding='utf-8') as fout:
                json.dump(type2id, fout)
        return examples


class MSRANerBaseProcessor:
    def __init__(self):
        self.type2id = {}

    # @staticmethod
    def read_json(self, file_path, set_type=None):
        examples = []
        type2id = {}
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
                    if label not in type2id:
                        type2id[label] = len(type2id)

        if set_type == 'train':
            self.type2id = type2id
            with open("type2id.json", 'w', encoding='utf-8') as fout:
                json.dump(type2id, fout)
        else:
            with open("type2id.json", 'r', encoding='utf-8') as fin:
                self.type2id = json.load(fin)
        return examples


class Processor(BaseProcessor):
    def __init__(self):
        super(BaseProcessor, self).__init__()
        pass

    # @staticmethod
    def _example_generator(self, raw_examples, set_type, max_seq_len):
        examples = []
        callback_info = []
        for _ex in raw_examples:
            text = _ex['text']
            labels = ["O" for _ in range(len(text))]
            entities = []
            for entity_type in _ex['label'].keys():
                entity_list = _ex['label'][entity_type]
                for entity_text in entity_list.keys():
                    entity_pos_list = entity_list[entity_text]
                    for pos_index in entity_pos_list:
                        entity_pos_start = pos_index[0]
                        entity_pos_end = pos_index[1] + 1
                        # print()
                        entities.append([entity_text, entity_type, entity_pos_start])
                        assert text[entity_pos_start: entity_pos_end] == entity_text
                        labels[entity_pos_start] = "B-" + entity_type
                        for i in range(entity_pos_start+1, entity_pos_end):
                            labels[i] = "I-" + entity_type
                        # labels.append([entity_text, entity_type, entity_pos_start, entity_pos_end])
            # print(text)
            # print(labels)
            # print(entities)
            examples.append(Example(set_type=set_type, text=text, label=labels))
            callback_info.append((text, entities))

        if set_type == 'dev':
            return examples, callback_info
        else:
            return examples

    def get_train_examples(self, raw_examples, max_seq_len):
        return self._example_generator(raw_examples, 'train', max_seq_len)

    def get_dev_examples(self, raw_examples, max_seq_len):
        return self._example_generator(raw_examples, 'dev', max_seq_len)

    def fine_grade_tokenize(self, raw_text, tokenizer):
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

    def convert_example(self, ex_idx, example: Example, max_seq_len, tokenizer: BertTokenizer):
        """
        convert attribution example to attribution feature
        """
        set_type = example.set_type
        raw_text = example.text
        raw_label = example.label

        tokens = self.fine_grade_tokenize(raw_text, tokenizer)
        assert len(tokens) == len(raw_text)
        labels = [0 for i in range(len(tokens))]  # start / end

        # tag labels
        for _index, _label in enumerate(raw_label):
            labels[_index] = self.type2id[_label]

        if len(labels) > max_seq_len - 2:
            labels = labels[:max_seq_len - 2]

        pad_labels = [0]
        labels = pad_labels + labels + pad_labels

        if len(labels) < max_seq_len:
            pad_length = max_seq_len - len(labels)
            labels = labels + pad_labels * pad_length

        assert len(labels) == max_seq_len

        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            is_split_into_words=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        token_ids = encode_dict['input_ids']
        # print(np.sum(np.array(token_ids)!=0), len(raw_text)+2)
        if len(raw_text) < max_seq_len - 2:
            assert np.sum(np.array(token_ids)!=0) == len(raw_text)+2

        attention_masks = encode_dict['attention_mask']
        token_type_ids = encode_dict['token_type_ids']
        # print(labels[: len(raw_text)+2])
        # print(raw_text)
        # print(raw_label)
        if ex_idx < 3 and set_type == 'train':
            logger.info(f"*** {set_type}_example-{ex_idx} ***")
            logger.info(f'text: {" ".join(tokens)}')
            logger.info(f"token_ids: {token_ids}")
            logger.info(f"attention_masks: {attention_masks}")
            logger.info(f"token_type_ids: {token_type_ids}")

        feature = Feature(token_ids=token_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          labels=labels)
        return feature

    def convert_examples_to_features(self, task_type, examples, bert_dir, max_seq_len):
        print(bert_dir)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=bert_dir, add_prefix_space=True)
        logger.info(f'Vocab nums in this tokenizer is: {tokenizer.vocab_size}')

        features = []

        for i, example in enumerate(tqdm(examples, desc=f'convert examples')):

            feature = self.convert_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer
            )

            if feature is None:
                continue

            features.append(feature)

        return features


class MSRANerProcessor(MSRANerBaseProcessor):
    def __init__(self):
        super(MSRANerBaseProcessor, self).__init__()
        pass

    def _example_generator(self, raw_examples, set_type, max_seq_len):
        examples = []
        callback_info = []
        for _ex in raw_examples[:1000]:
            text = _ex['text']
            labels = _ex['labels']
            entities = []
            entity_text = ""
            entity_type = None
            entity_start = None
            # entity_end = None
            for i in range(len(labels)):
                if labels[i][0] == 'B':
                    entity_text += text[i]
                    entity_type = labels[i][2:]
                    entity_start = i
                if labels[i][0] == 'M':
                    entity_text += text[i]
                if labels[i][0] == "E":
                    entity_text += text[i]
                    # entity_end = i+1
                    entities.append([entity_text, entity_type, entity_start])
                    entity_text = ""
                    entity_type = None
                    entity_start = None

            examples.append(Example(set_type=set_type, text=text, label=labels))
            callback_info.append((text, entities))

        if set_type == 'dev':
            return examples, callback_info
        else:
            return examples

    def get_train_examples(self, raw_examples, max_seq_len):
        return self._example_generator(raw_examples, 'train', max_seq_len)

    def get_dev_examples(self, raw_examples, max_seq_len):
        return self._example_generator(raw_examples, 'dev', max_seq_len)

    def fine_grade_tokenize(self, raw_text, tokenizer):
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

    def convert_example(self, ex_idx, example: Example, max_seq_len, tokenizer: BertTokenizer):
        """
        convert attribution example to attribution feature
        """
        set_type = example.set_type
        raw_text = example.text
        raw_label = example.label

        tokens = self.fine_grade_tokenize(raw_text, tokenizer)
        assert len(tokens) == len(raw_text)
        labels = [0 for i in range(len(tokens))]  # start / end

        # tag labels
        for _index, _label in enumerate(raw_label):
            labels[_index] = self.type2id[_label]

        if len(labels) > max_seq_len - 2:
            labels = labels[:max_seq_len - 2]

        pad_labels = [0]
        labels = pad_labels + labels + pad_labels

        if len(labels) < max_seq_len:
            pad_length = max_seq_len - len(labels)
            labels = labels + pad_labels * pad_length

        assert len(labels) == max_seq_len

        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            is_split_into_words=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        token_ids = encode_dict['input_ids']
        # print(np.sum(np.array(token_ids)!=0), len(raw_text)+2)
        if len(raw_text) < max_seq_len - 2:
            assert np.sum(np.array(token_ids)!=0) == len(raw_text)+2

        attention_masks = encode_dict['attention_mask']
        token_type_ids = encode_dict['token_type_ids']
        # print(labels[: len(raw_text)+2])
        # print(raw_text)
        # print(raw_label)
        if ex_idx < 3 and set_type == 'train':
            logger.info(f"*** {set_type}_example-{ex_idx} ***")
            logger.info(f'text: {" ".join(tokens)}')
            logger.info(f"token_ids: {token_ids}")
            logger.info(f"attention_masks: {attention_masks}")
            logger.info(f"token_type_ids: {token_type_ids}")

        feature = Feature(token_ids=token_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          labels=labels)
        return feature

    def convert_examples_to_features(self, task_type, examples, bert_dir, max_seq_len):
        print(bert_dir)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=bert_dir, add_prefix_space=True)
        logger.info(f'Vocab nums in this tokenizer is: {tokenizer.vocab_size}')

        features = []

        for i, example in enumerate(tqdm(examples, desc=f'convert examples')):

            feature = self.convert_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer
            )

            if feature is None:
                continue

            features.append(feature)

        return features