# coding=utf-8
import os
import json
import torch
import torch.nn as nn
from pytorchcrf import CRF
from transformers import BertModel, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import BertTokenizer, AutoTokenizer
import numpy as np


class SeviceModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob, k=2):
        super(SeviceModel, self).__init__()
        bert_module = AutoModel.from_pretrained(bert_dir)
        self.service_model = bert_module
        self.service_model.encoder.layer = self.service_model.encoder.layer[k:]
        # print(self.service_model)
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.activation = nn.Sigmoid()
        self.criterion = nn.CrossEntropyLoss()

    def forward_bert(self, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None,
                    position_ids=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                    past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.service_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.service_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.service_model.config.use_return_dict

        if self.service_model.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.service_model.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.service_model.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.service_model.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.service_model.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.service_model.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.service_model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.service_model.get_head_mask(head_mask, self.service_model.config.num_hidden_layers)

        encoder_outputs = self.service_model.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.service_model.pooler(sequence_output) if self.service_model.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def forward(self, inputs_embeds=None, attention_mask=None, token_type_ids=None, labels=None):
        seq_out = self.forward_bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
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
