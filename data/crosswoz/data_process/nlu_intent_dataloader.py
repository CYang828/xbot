import numpy as np
import torch
import random
from transformers import BertTokenizer
import math
from collections import Counter


class Dataloader:
    def __init__(self, intent_vocab, pretrained_weights):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param pretrained_weights: which bert_policy, e.g. 'bert_policy-base-uncased'
        """
        self.intent_vocab = intent_vocab  # len(intent_vocab)
        self.intent_dim = len(intent_vocab)
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)

    def load_data(self, data, data_key, cut_sen_len, use_bert_tokenizer=True):
        """
        cut_sen_len参数在配置文件里进行配置
        use_bert_tokenizer 参数在配置文件里
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param use_bert_tokenizer:
        :param cut_sen_len:
        :param data_key: train/val/tests
        :param data:
        :return:
        """
        self.data[data_key] = data
        max_sen_len, max_context_len = 0, 0
        sen_len = []  # 存放所有数据的长度

        for d in self.data[data_key]:
            max_sen_len = max(max_sen_len, len(d[0]))
            sen_len.append(len(d[0]))
            if cut_sen_len > 0:  # 对每条数据进行长度的截取
                d[0] = d[0][:cut_sen_len]
            if use_bert_tokenizer:
                word_seq = self.bert_tokenize(d[0])
            else:
                word_seq = d[0]
            d.append(word_seq)
            d.append(self.seq_intent2id(d[1]))
            # [list of words, list of tags, list of intents, original dialog act,问句，list of words, intent id]
            if data_key == "train":
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
        if data_key == "train":
            train_size = len(self.data["train"])
            for intent, intent_id in self.intent2id.items():
                neg_pos = (
                    train_size - self.intent_weight[intent_id]
                ) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
            self.intent_weight = torch.tensor(self.intent_weight)
        print("max sen bert_policy len", max_sen_len)
        print(sorted(Counter(sen_len).items()))

    def bert_tokenize(self, word_seq):
        split_tokens = []
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
                split_tokens.append(sub_token)
            if accum == "":
                j += 1
        return split_tokens

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]

    def pad_batch(self, batch_data):
        batch_size = len(batch_data)
        max_seq_len = max([len(x[0]) for x in batch_data]) + 2
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)  #

        for i in range(batch_size):
            words = batch_data[i][0]
            intents = batch_data[i][-1]
            words = ["[CLS]"] + words + ["[SEP]"]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(words)
            sen_len = len(words)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)

            for j in intents:
                intent_tensor[i, j] = 1.0

        return word_seq_tensor, word_mask_tensor, intent_tensor

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data["train"], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size : (i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
