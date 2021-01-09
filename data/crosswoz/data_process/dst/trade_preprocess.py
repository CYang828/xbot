import os
import bz2
import json
import random
import pickle
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

EXPERIMENT_DOMAINS = ["餐馆", "地铁", "出租", "酒店", "景点"]


class Lang:
    def __init__(self, config):
        self.index2word = {
            config["pad_id"]: "PAD",
            config["sos_id"]: "SOS",
            config["eos_id"]: "EOS",
            config["unk_id"]: "UNK",
        }
        self.word2index = {v: k for k, v in self.index2word.items()}
        self.n_words = len(self.index2word)  # Count default tokens

    def index_words(self, sent, type_):
        if type_ == "utter":
            for word in sent.split(" "):
                self.index_word(word)
        elif type_ == "slot":
            for slot in sent:
                items = slot.split("-", 1)
                assert len(items) == 2, slot
                d, s = items
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
        elif type_ == "belief":
            for slot, value in sent.items():
                # d-domain
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def fix_general_label_error(belief_state):
    """
    :param belief_state:
        "belief_state": [
          {
            "slots": [
              [
                "餐馆-推荐菜",
                "驴 杂汤"
              ]
            ]
          },
          {
            "slots": [
              [
                "餐馆-人均消费",
                "100 - 150 元"
              ]
            ]
          }
        ]
    :return:
    """
    belief_state_dict = {
        slot_value["slots"][0][0]: slot_value["slots"][0][1]
        for slot_value in belief_state
    }
    return belief_state_dict


class CNEmbedding:
    def __init__(self):
        vector_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "crosswoz",
            "sgns.wiki.bigram.bz2",
        )
        self.word2vec = {}
        with bz2.open(vector_path, "rt", encoding="utf8") as fin:
            lines = fin.readlines()
            # 第一行是元信息
            for line in tqdm(lines[1:], desc="Generating pretrained embeddings"):
                line = line.strip()
                tokens = line.split()
                word = tokens[0]
                vec = tokens[1:]
                vec = [float(item) for item in vec]
                self.word2vec[word] = vec
        self.embed_size = 300

    def emb(self, token, default="zero"):
        get_default = {
            "none": lambda: None,
            "zero": lambda: 0.0,
            "random": lambda: random.uniform(-0.1, 0.1),
        }[default]

        vec = self.word2vec.get(token, None)
        if vec is None:
            vec = [get_default()] * self.embed_size
        return vec


class CrossWOZDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id, mem_word2id, config):
        """Reads source and target sequences from txt files."""
        self.ID = data_info["ID"]
        self.turn_domain = data_info["turn_domain"]
        self.turn_id = data_info["turn_id"]
        self.dialog_history = data_info["dialog_history"]
        self.turn_belief = data_info["turn_belief"]
        self.gating_label = data_info["gating_label"]
        self.turn_uttr = data_info["turn_uttr"]
        self.generate_y = data_info["generate_y"]  # slot 的 value 值
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id  # utterance 的词表
        self.mem_word2id = mem_word2id  # slot-value 的词表
        self.config = config

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        id_ = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])
        generate_y = self.generate_y[index]
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
        context = self.dialog_history[index]
        context = self.preprocess(context, self.src_word2id)  # id
        context_plain = self.dialog_history[index]  # 文本

        item_info = {
            "ID": id_,
            "turn_id": turn_id,
            "turn_belief": turn_belief,
            "gating_label": gating_label,
            "context": context,
            "context_plain": context_plain,
            "turn_uttr_plain": turn_uttr,
            "turn_domain": turn_domain,
            "generate_y": generate_y,
        }
        return item_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [
            word2idx[word] if word in word2idx else self.config["unk_id"]
            for word in sequence.split()
        ]
        story = torch.tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            # sequence 是 decoder 需要生成的 value 的真实值，所以需要加上 eos
            v = [
                word2idx[word] if word in word2idx else self.config["unk_id"]
                for word in value.split()
            ] + [self.config["eos_id"]]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    @staticmethod
    def preprocess_domain(turn_domain):
        """将 domain text 变成 id"""
        domains = {"餐馆": 0, "地铁": 1, "出租": 2, "酒店": 3, "bye": 4, "景点": 5, "greet": 6}
        if turn_domain not in domains:
            # 直接获取第一个？这是什么做法
            key = list(domains.keys())[0]
            return domains[key]
        return domains[turn_domain]


def collate_fn(data, pad_id=1, n_gpus=0):
    def merge(sequences):
        """merge from batch * sent_len to batch * max_len
        :param sequences:
        :return:
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()  # pad_id=1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach()  # torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        """merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        :param sequences:
        :return:
        """
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(length) for length in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [pad_id] * (max_len - len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x["context"]), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info["context"])  # (bs, max_len)
    y_seqs, y_lengths = merge_multi_response(
        item_info["generate_y"]
    )  # (bs, num_slots, max_len)
    gating_label = torch.tensor(item_info["gating_label"])  # (bs, num_slots)
    turn_domain = torch.tensor(item_info["turn_domain"])  # (bs, )

    if n_gpus > 0:
        src_seqs = src_seqs.cuda()
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()
        src_lengths = src_lengths.cuda()

    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths
    return item_info


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        super(ImbalancedDatasetSampler, self).__init__(dataset)
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            labels = self._get_label(dataset, idx)
            for label in labels:
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0
            / sum([label_to_count[label] for label in self._get_label(dataset, idx)])
            for idx in self.indices
        ]
        self.weights = torch.tensor(weights, dtype=torch.float64)

    @staticmethod
    def _get_label(dataset, idx):
        return dataset.gating_label[idx]

    def __iter__(self):
        return (
            self.indices[i.item()]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


def get_slot_information(ontology):
    slots = [k for k, _ in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS]
    return slots


def read_langs_for_update(source_text, utterance):
    data = []
    turn_domain = ""
    turn_id = "0"
    dialogue_idx = "0"
    turn_uttr = utterance
    turn_uttr_strip = turn_uttr.strip()
    source_text = source_text.strip()

    gating_label = [1] * 80
    generate_y = ["none"] * 80

    # 可以根据ID和turn_idx将内容复原
    data_detail = {
        "ID": dialogue_idx,
        "domains": [],
        "turn_domain": turn_domain,
        "turn_id": turn_id,
        "dialog_history": source_text,
        "turn_belief": [],
        "gating_label": gating_label,
        "turn_uttr": turn_uttr_strip,
        "generate_y": generate_y,
    }
    data.append(data_detail)

    return data


def get_seq(pairs, lang, mem_lang, batch_size, n_gpus, shuffle, config):
    if shuffle:
        random.shuffle(pairs)

    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []

    # 将每个 sample 中的键值按键存储，不按照 sample 存储了，
    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    # 这里对于 generate_y 的转换是不是要使用 mem_lang.word2index
    dataset = CrossWOZDataset(
        data_info, lang.word2index, lang.word2index, mem_lang.word2index, config
    )
    collate = partial(collate_fn, pad_id=config["pad_id"], n_gpus=n_gpus)

    if config["use_imbalance_sampler"] and shuffle:
        print("Using Imbalanced Dataset Sampler")
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate,
            sampler=ImbalancedDatasetSampler(dataset),
        )
    else:
        data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
        )
    return data_loader


def prepare_data_for_update(
    config, lang, mem_lang, batch_size=100, source_text="", curr_utterance=""
):

    utterance = curr_utterance
    if source_text == "":
        source_text = "; " + utterance.strip() + " ;"
    pair_test = read_langs_for_update(source_text, utterance)
    data_loader = get_seq(
        pair_test,
        lang,
        mem_lang,
        batch_size,
        config["n_gpus"],
        shuffle=False,
        config=config,
    )

    return data_loader
