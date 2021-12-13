# coding=utf-8
import os
import json
import torch
import torch.nn as nn
from pytorchcrf import CRF
from transformers import BertModel, AutoModel
from transformers import BertTokenizer, AutoTokenizer
import numpy as np
from Client.model_utils import ClientModel
from Service.model_utils import SeviceModel
import copy


if __name__ == '__main__':
    bert_dir = "/Users/jiezhou/Desktop/其他/张旗/bert_embedding/BERT_models/"
    client_model = ClientModel(bert_dir=bert_dir, dropout_prob=0.1, k=2)
    tokenizer = AutoTokenizer.from_pretrained(bert_dir, add_prefix_space=True)
    raw_text_original = "I love you !"
    max_seq_len = 10
    encode_dict = tokenizer.encode_plus(text=raw_text_original,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        # is_split_into_words=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    # print(encode_dict)
    token_ids = torch.Tensor(np.array([encode_dict['input_ids']], dtype='int64')).long()
    attention_masks = torch.Tensor(np.array([encode_dict['attention_mask']], dtype='int64')).long()
    token_type_ids = torch.Tensor(np.array([encode_dict['token_type_ids']], dtype='int64')).long()

    client_output = client_model(token_ids, attention_masks, token_type_ids)

    # print(output[0][0])
    input = client_output[0].detach().clone()#.requires_grad_(True)
    input.requires_grad = True
    # input = output[0][0]
    # print(input)
    service_model = SeviceModel(bert_dir=bert_dir, dropout_prob=0.1, k=2)
    labels = torch.ones(attention_masks.size()[0], attention_masks.size()[1]).view(-1).long()
    service_output = service_model(inputs_embeds=input, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
    # print(output[1])
    loss_service = service_output[1]
    loss_service.backward()
    # print(output[0])

    gradient = input.grad.data.clone()
    # print(gradient.size(), client_output[0].size())
    loss_client = torch.mean(client_output[0]*gradient)
    print(loss_client, loss_service)
    loss_client.backward()
