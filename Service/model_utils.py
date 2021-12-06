# coding=utf-8
import os
import json
import torch
import torch.nn as nn
from pytorchcrf import CRF
from transformers import BertModel, AutoModel
from transformers import BertTokenizer, AutoTokenizer
import numpy as np


class SeviceModel(nn.Module):
    def __init__(self,  bert_dir, dropout_prob, k=2):
        super(SeviceModel, self).__init__()
        bert_module = AutoModel.from_pretrained(bert_dir)
        self.service_model=bert_module
        self.service_model.encoder.layer = self.service_model.encoder.layer[k:]
        # print(self.service_model)
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.activation = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None):
        seq_out = self.service_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        logits = self.classifier(self.dropout_layer(seq_out))
        out = (logits,)
        if labels is not None:
            masks = torch.unsqueeze(attention_mask, -1)
            # labels = labels.float()
            # print((logits*masks).size())
            predicted = logits * masks
            predicted = predicted.view(-1, predicted.size(-1))
            loss = self.criterion(predicted, labels)
            out = out + (loss,)
        return out
