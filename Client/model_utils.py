# coding=utf-8
import os
import json
import torch
import torch.nn as nn
from pytorchcrf import CRF
from transformers import BertModel, AutoModel
from transformers import BertTokenizer, AutoTokenizer
import numpy as np


class ClientModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob, k=2):
        super(ClientModel, self).__init__()
        bert_module = AutoModel.from_pretrained(bert_dir)
        self.client_model = bert_module
        self.client_model.encoder.layer = self.client_model.encoder.layer[: k]
        self.dropout_layer = nn.Dropout(dropout_prob)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        seq_output = self.client_model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]
        out = (seq_output, )
        if labels is not None:
            loss = torch.sum(labels * seq_output)
            return out + (loss, )
        return out


class ClientCompressModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob, k=2):
        super(ClientCompressModel, self).__init__()
        bert_module = AutoModel.from_pretrained(bert_dir)
        self.client_model = bert_module
        self.client_model.encoder.layer = self.client_model.encoder.layer[: k]
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.down_layer = nn.Linear(768, 128)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        seq_output = self.client_model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]
        seq_output = self.down_layer(self.dropout_layer(seq_output))
        out = (seq_output, )
        if labels is not None:
            loss = torch.sum(labels * seq_output)
            return out + (loss, )
        return out