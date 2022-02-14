# coding=utf-8
import os
import json
import torch
import torch.nn as nn
from pytorchcrf import CRF
from transformers import BertModel, AutoModel
from transformers import BertTokenizer, AutoTokenizer

import numpy as np


class BertNERModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob):
        super(BertNERModel, self).__init__()
        bert_module = AutoModel.from_pretrained(bert_dir)
        self.bert_module = bert_module
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(768, 13)
        self.activation = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        seq_output = self.bert_module(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)[0]
        logits = self.classifier(self.dropout_layer(seq_output))
        out = (logits,)
        if labels is not None:
            masks = torch.unsqueeze(attention_mask, -1)
            # labels = labels.float()
            # print((logits*masks).size())
            predicted = logits * masks
            predicted = predicted.view(-1, predicted.size(-1))
            loss = self.criterion(predicted, labels.view(-1))
            out = out + (loss,)
        return out
