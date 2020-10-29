# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 10:34 上午
# @Author  : zhengjiawei
# @FileName: slot_bert_model.py
# @Software: PyCharm

import torch
from torch import nn
from transformers import BertModel


class JointBERT(nn.Module):
    def __init__(self, model_config, device, slot_dim):
        super(JointBERT, self).__init__()
        self.slot_num_labels = slot_dim
        self.device = device
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:#选择神经元不同的全联接层
            if self.context:
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.slot_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
            else:
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.slot_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
            nn.init.xavier_uniform_(self.slot_hidden.weight)#初始化参数，服从均匀分布
        else:
            if self.context:
                self.slot_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.slot_num_labels)
            else:
                self.slot_classifier = nn.Linear(self.bert.config.hidden_size, self.slot_num_labels)
        nn.init.xavier_uniform_(self.slot_classifier.weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                context_seq_tensor=None, context_mask_tensor=None):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0] #输入的是word_seq_tensor，bert的输出有两部分，这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
        pooled_output = outputs[1]#这个输出 是获取句子的output

        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                #输入context_seq_tensor，同样得到的输出有两个，取第二个
                context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)
        pooled_output = self.dropout(pooled_output)
        outputs= outputs
        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)
            outputs = outputs + (slot_loss,)
        return outputs
