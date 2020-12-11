import os
import bz2
import json
import random
import pickle
from collections import defaultdict, Counter

from tqdm import tqdm

import torch

from data.crosswoz.data_process.dst.trade_preprocess import (
    EXPERIMENT_DOMAINS,
    Lang,
    get_seq,
    get_slot_information,
)


class CNEmbedding:
    def __init__(self, vector_path):
        self.word2vec = {}
        with bz2.open(vector_path, "rt", encoding="utf8") as fin:
            lines = fin.readlines()
            # 第一行是元信息
            for line in tqdm(lines[1:], desc="Generating pretrained embedding"):
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


def dump_pretrained_emb(orig_embedding_path, index2word, dump_path):
    print("Dumping pretrained embeddings...")

    embeddings = [CNEmbedding(orig_embedding_path)]
    embedding = []
    count = [0.0, 0.0]
    for i in tqdm(range(len(index2word))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default="zero")

        # stat embed existence
        count[1] += 1.0  # 总词数
        if w in embeddings[0].word2vec:
            count[0] += 1.0  # 存在于 embedding 中的词数
        # e += [0.] * 300
        embedding.append(e)

    with open(dump_path, "w") as f:
        json.dump(embedding, f)
    print(f"Word exists in embedding mat: {count[0] / count[1] * 100}")


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


def read_langs(
    file_name, gating_dict, slots, dataset, lang, mem_lang, load_lang, config
):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = defaultdict(int)  # 每个 domain 有多少个
    gate_counter = []
    with open(file_name, "r", encoding="utf8") as f:
        dials = json.load(f)
        if config["debug"]:
            dials = dials[:10]

        # create vocab first
        for dial_dict in dials:  # 一个 dial_dict 就是一个对话，包括多轮
            if not load_lang and (config["all_vocab"] or dataset == "train"):
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    # 生成 utterance 的词表
                    lang.index_words(turn["system_transcript"], "utter")
                    lang.index_words(turn["transcript"], "utter")

        for dial_dict in dials:
            dialog_history = ""
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                domain_counter[domain] += 1

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]  # 数据集里都是 0，好像有问题
                turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                turn_uttr_strip = turn_uttr.strip()
                dialog_history += (
                    turn["system_transcript"] + " ; " + turn["transcript"] + " ; "
                )
                source_text = dialog_history.strip()
                # 用于英文数据集， {"餐馆-推荐菜": "驴 杂汤"} dev_dials.json 第一个
                turn_belief_dict = fix_general_label_error(turn["belief_state"])

                # List['domain-slot-value']
                turn_belief_list = [
                    str(k) + "-" + str(v) for k, v in turn_belief_dict.items()
                ]

                if not load_lang and (config["all_vocab"] or dataset == "train"):
                    # 生成 slot-value 的词表
                    mem_lang.index_words(turn_belief_dict, "belief")

                class_label, generate_y, slot_mask, gating_label = [], [], [], []
                # 一个轮次的 slot 的 values 和 ontology 中的数量一样多
                for slot in slots:  # ontology
                    # 只关注本轮需要的 ontology
                    if slot in turn_belief_dict.keys():  # dialogue
                        generate_y.append(turn_belief_dict[slot])

                        # ontology 中是有 none 的情况的
                        if turn_belief_dict[slot] == "none":  # none 存在也只能是下面那种情况
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])

                gate_counter.extend(gating_label)

                # 可以根据ID和turn_idx将内容复原
                data_detail = {
                    "ID": dial_dict["dialogue_idx"],
                    "domains": dial_dict["domains"],
                    "turn_domain": turn_domain,
                    "turn_id": turn_id,  # 好像都是 0
                    "dialog_history": source_text,
                    "turn_belief": turn_belief_list,
                    "gating_label": gating_label,
                    "turn_uttr": turn_uttr_strip,  # 每一轮的系统和人的话语
                    "generate_y": generate_y,
                }
                data.append(data_detail)

                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())  # 对话数量，系统和人各算一个

    # add t{} to the lang file 用来干啥的
    if "t{}".format(max_value_len - 1) not in mem_lang.word2index.keys():
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), "utter")

    print("domain_counter", domain_counter)
    print("gate counter", Counter(gate_counter))
    return data, max_resp_len


def prepare_data_seq(config):
    eval_batch = (
        config["eval_batch_size"] if config["eval_batch_size"] else config["batch_size"]
    )
    train_file_path = config["train_dials"]
    dev_file_path = config["dev_dials"]
    test_file_path = config["test_dials"]
    ontology_file_path = config["ontology"]

    # load domain-slot pairs from ontology
    ontology = json.load(open(ontology_file_path, "r", encoding="utf8"))
    slots = get_slot_information(ontology)
    gating_dict = {"ptr": 0, "none": 1}

    # Vocabulary
    lang_name = "lang-all.pkl" if config["all_vocab"] else "lang-train.pkl"
    mem_lang_name = "mem-lang-all.pkl" if config["all_vocab"] else "mem-lang-train.pkl"
    if config["debug"]:
        lang_name = "debug-" + lang_name
        mem_lang_name = "debug-" + mem_lang_name
    lang_file_path = os.path.join(config["data_path"], lang_name)
    mem_lang_file_path = os.path.join(config["data_path"], mem_lang_name)
    load_lang = False
    if (
        os.path.exists(lang_file_path) and os.path.exists(mem_lang_file_path)
    ) and not config["clean_cache"]:
        print("Loading saved lang files...")
        load_lang = True
        with open(lang_file_path, "rb") as f:
            lang = pickle.load(f)
        with open(mem_lang_file_path, "rb") as f:
            mem_lang = pickle.load(f)
    else:
        lang, mem_lang = Lang(config), Lang(config)
        # 都包含了 ontology 中的 domain 和 slot，之后分别包含 utterance 和 domain-slot-value
        lang.index_words(slots, "slot")
        mem_lang.index_words(slots, "slot")

    # 生成 dataloader
    pair_train, train_max_len = read_langs(
        train_file_path, gating_dict, slots, "train", lang, mem_lang, load_lang, config
    )
    train_loader = get_seq(
        pair_train,
        lang,
        mem_lang,
        config["batch_size"],
        config["n_gpus"],
        shuffle=True,
        config=config,
    )
    train_vocab_size = lang.n_words

    pair_dev, dev_max_len = read_langs(
        dev_file_path, gating_dict, slots, "dev", lang, mem_lang, load_lang, config
    )
    dev_loader = get_seq(
        pair_dev,
        lang,
        mem_lang,
        eval_batch,
        config["n_gpus"],
        shuffle=False,
        config=config,
    )

    pair_test, test_max_len = read_langs(
        test_file_path, gating_dict, slots, "tests", lang, mem_lang, load_lang, config
    )
    test_loader = get_seq(
        pair_test,
        lang,
        mem_lang,
        eval_batch,
        config["n_gpus"],
        shuffle=False,
        config=config,
    )

    # 保存中间数据
    if (
        not (os.path.exists(lang_file_path) and os.path.exists(mem_lang_file_path))
        or config["clean_cache"]
    ):
        print("Dumping lang files...")
        with open(lang_file_path, "wb") as f:
            pickle.dump(lang, f)
        with open(mem_lang_file_path, "wb") as f:
            pickle.dump(mem_lang, f)

    emb_dump_path = os.path.join(config["data_path"], f"emb{len(lang.index2word)}")
    if (not os.path.exists(emb_dump_path) or config["clean_cache"]) and config[
        "load_embedding"
    ]:
        dump_pretrained_emb(
            config["orig_pretrained_embedding"], lang.index2word, emb_dump_path
        )

    max_dialogue_history_length = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs tests" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % train_vocab_size)
    print("Vocab_size Belief %s" % mem_lang.n_words)
    print("Max. length of dialog words for RNN: %s " % max_dialogue_history_length)

    langs = [lang, mem_lang]
    # dataloader, dataloader, dataloader, dataloader, List[Lang], List[Dict[str, str]], Dict[str, int], int
    return train_loader, dev_loader, test_loader, langs, slots, gating_dict


def masked_cross_entropy_for_value(logits, target, mask):
    # logits: b * |s| * m * |v|
    # target: b * |s| * m
    # mask:   b * |s|
    logits_flat = logits.view(-1, logits.size(-1))
    # print(logits_flat.size())
    log_probs_flat = torch.log(logits_flat)
    # print("log_probs_flat", log_probs_flat)
    target_flat = target.view(-1, 1)
    # print("target_flat", target_flat)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())  # b * |s| * m
    loss = masking(losses, mask)
    return loss


def masking(losses, mask):
    mask_ = []
    batch_size = mask.size(0)
    max_len = losses.size(2)
    for si in range(mask.size(1)):
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        if mask[:, si].is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (
            mask[:, si].unsqueeze(1).expand_as(seq_range_expand)
        )  # (bs, max_len)
        mask_.append((seq_range_expand < seq_length_expand))
    mask_ = torch.stack(mask_)
    mask_ = mask_.transpose(0, 1)  # (bs, num_slots, max_len)
    if losses.is_cuda:
        mask_ = mask_.cuda()
    losses = losses * mask_.float()
    loss = losses.sum() / (mask_.sum().float())
    return loss


def reformat_belief_state(raw_state):
    belief_state = []
    for item in raw_state:
        dsv_triple = item.split("-", 2)
        domain = dsv_triple[0].strip()
        slot = dsv_triple[1].strip()
        value = dsv_triple[2].strip()
        belief_state.append({"slots": [[domain + "-" + slot, value]]})
    return belief_state


def compute_acc(gold, pred, slot_temp):
    # TODO 为什么不求交集直接算
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])  # g=domain-slot-value
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    acc_total = len(slot_temp)
    # slot_temp 包含所有 80 个 domain-slot，一轮对话总共可能就几个，这么算不合适吧
    acc = len(slot_temp) - miss_gold - wrong_pred
    acc = acc / float(acc_total)
    return acc


def compute_prf(gold, pred):
    tp, fp, fn = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                tp += 1
            else:
                fn += 1
        for p in pred:
            if p not in gold:
                fp += 1
        precision = tp / float(tp + fp) if (tp + fp) != 0 else 0
        recall = tp / float(tp + fn) if (tp + fn) != 0 else 0
        f1 = (
            2 * precision * recall / float(precision + recall)
            if (precision + recall) != 0
            else 0
        )
    else:
        if not pred:
            precision, recall, f1, count = 1, 1, 1, 1
        else:
            precision, recall, f1, count = 0, 0, 0, 1
    return f1, recall, precision, count


def evaluate_metrics(all_prediction, from_which, slot_temp):
    total, turn_acc, joint_acc, f1_pred, f1_count = 0, 0, 0, 0, 0
    for d, v in all_prediction.items():
        for t in range(len(v)):
            cv = v[t]
            if set(cv["turn_belief"]) == set(cv[from_which]):
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(
                set(cv["turn_belief"]), set(cv[from_which]), slot_temp
            )
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(
                set(cv["turn_belief"]), set(cv[from_which])
            )
            f1_pred += temp_f1
            f1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    turn_acc_score = turn_acc / float(total) if total != 0 else 0
    f1_score = f1_pred / float(f1_count) if f1_count != 0 else 0
    return joint_acc_score, f1_score, turn_acc_score
