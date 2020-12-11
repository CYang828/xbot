# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 11:29 上午
# @Author  : zhengjiawei
# @FileName: nlu_slot_dataloader.py
# @Software: PyCharm

import torch
import random
from transformers import BertTokenizer
import math


class Dataloader:
    def __init__(self, intent_vocab, tag_vocab, pretrained_weights):
        self.tag_vocab = tag_vocab
        self.intent_dim = len(intent_vocab)
        self.tag_dim = len(tag_vocab)
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)

    def load_data(self, data, data_key, cut_sen_len, use_bert_tokenizer=True):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/tests
        :param data:
        :return:
        """
        self.data[data_key] = data
        max_sen_len, max_context_len = 0, 0
        sen_len = []
        context_len = []
        for d in self.data[data_key]:
            max_sen_len = max(max_sen_len, len(d[0]))  # 计算最大句子长度
            sen_len.append(len(d[0]))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]
                d[1] = d[1][:cut_sen_len]
                d[3] = [" ".join(s.split()[:cut_sen_len]) for s in d[3]]

            d[3] = self.tokenizer.encode("[CLS] " + " [SEP] ".join(d[3]))
            max_context_len = max(max_context_len, len(d[3]))
            context_len.append(len(d[3]))

            if use_bert_tokenizer:
                word_seq, tag_seq, new2ori = self.bert_tokenize(d[0], d[1])
            else:
                word_seq = d[0]
                tag_seq = d[1]
                new2ori = None
            d.append(new2ori)
            d.append(word_seq)
            d.append(self.seq_tag2id(tag_seq))

    def bert_tokenize(self, word_seq, tag_seq):
        split_tokens = []
        new_tag_seq = []
        new2ori = {}
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(" ".join(word_seq))
        accum = ""
        i, j = 0, 0
        for i, token in enumerate(basic_tokens):
            if (accum + token).lower() == word_seq[j].lower():
                accum = ""
            else:
                accum += token
            for sub_token in self.tokenizer.wordpiece_tokenizer.tokenize(
                basic_tokens[i]
            ):
                new2ori[len(new_tag_seq)] = j
                split_tokens.append(sub_token)
                new_tag_seq.append(tag_seq[j])
            if accum == "":
                j += 1
        return split_tokens, new_tag_seq, new2ori

    def seq_tag2id(self, tags):
        return [self.tag2id[x] for x in tags if x in self.tag2id]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]

    def pad_batch(self, batch_data):
        batch_size = len(batch_data)
        max_seq_len = max([len(x[0]) for x in batch_data]) + 2
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        context_max_seq_len = max([len(x[3]) for x in batch_data])
        context_mask_tensor = torch.zeros(
            (batch_size, context_max_seq_len), dtype=torch.long
        )
        context_seq_tensor = torch.zeros(
            (batch_size, context_max_seq_len), dtype=torch.long
        )
        for i in range(batch_size):
            words = batch_data[i][-2]
            tags = batch_data[i][-1]
            words = ["[CLS]"] + words + ["[SEP]"]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(words)
            sen_len = len(words)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
            tag_seq_tensor[i, 1 : sen_len - 1] = torch.LongTensor(tags)
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            tag_mask_tensor[i, 1 : sen_len - 1] = torch.LongTensor([1] * (sen_len - 2))
            context_len = len(batch_data[i][3])

            context_seq_tensor[i, :context_len] = torch.LongTensor([batch_data[i][3]])
            context_mask_tensor[i, :context_len] = torch.LongTensor([1] * context_len)

        return (
            word_seq_tensor,
            tag_seq_tensor,
            word_mask_tensor,
            tag_mask_tensor,
            context_seq_tensor,
            context_mask_tensor,
        )

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data["train"], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size : (i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
