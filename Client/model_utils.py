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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None):
        seq_output = self.client_model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        out = (seq_output, )
        if label is not None:
            loss = torch.sum(label * seq_output)
            return out + loss
        return out
