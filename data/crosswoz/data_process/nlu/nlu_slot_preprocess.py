import json
import os
import zipfile
import sys
from collections import Counter
from transformers import BertTokenizer


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, "r")
    return json.load(archive.open(filename))


def preprocess(mode):
    ##model =='all' 表示同时使用用户话语和系统话语进行训练，
    # 如果** mode ** ='usr'，则使用用户话语进行训练；如果** mode ** ='sys'，则使用系统话语进行训练；

    assert mode == "all" or mode == "usr" or mode == "sys"
    cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(cur_dir, "raw")
    processed_data_dir = os.path.join(cur_dir, "slot_temp_data/{}_data".format(mode))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ["train", "val", "tests"]
    data = {}
    ##导入train，tests,val文件
    for key in data_key:
        data[key] = read_zipped_json(
            os.path.join(data_dir, key + ".json.zip"), key + ".json"
        )
        print("load {}, size {}".format(key, len(data[key])))
    processed_data = {}
    all_intent = []
    all_tag = []

    context_size = 3
    # BERT-wwm-ext是哈工大讯飞联合实验室发布的中文预训练模型
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert_policy-wwm-ext")

    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            context = []
            for i, turn in enumerate(sess["messages"]):
                if mode == "usr" and turn["role"] == "sys":
                    context.append(turn["content"])
                    continue
                elif mode == "sys" and turn["role"] == "usr":
                    context.append(turn["content"])
                    continue
                utterance = turn["content"]
                tokens = tokenizer.tokenize(utterance)
                golden = []

                span_info = []
                intents = []
                for intent, domain, slot, value in turn["dialog_act"]:
                    if intent in ["Inform", "Recommend"] and "酒店设施" not in slot:
                        if value in utterance:
                            idx = utterance.index(value)
                            idx = len(tokenizer.tokenize(utterance[:idx]))
                            span_info.append(
                                (
                                    "+".join([intent, domain, slot]),
                                    idx,
                                    idx + len(tokenizer.tokenize(value)),
                                    value,
                                )
                            )
                            token_v = "".join(
                                tokens[idx : idx + len(tokenizer.tokenize(value))]
                            )
                            token_v = token_v.replace("##", "")
                            golden.append([intent, domain, slot, token_v])
                        else:
                            golden.append([intent, domain, slot, value])
                    else:
                        intents.append("+".join([intent, domain, slot, value]))
                        golden.append([intent, domain, slot, value])

                tags = []
                for j, _ in enumerate(tokens):
                    for span in span_info:
                        if j == span[1]:
                            tag = "B+" + span[0]
                            tags.append(tag)
                            break
                        if span[1] < j < span[2]:
                            tag = "I+" + span[0]
                            tags.append(tag)
                            break
                    else:
                        tags.append("O")

                processed_data[key].append(
                    [tokens, tags, golden, context[-context_size:]]
                )

                all_intent += intents
                all_tag += tags

                context.append(turn["content"])

        all_intent = [x[0] for x in dict(Counter(all_intent)).items()]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items()]
        print("loaded {}, size {}".format(key, len(processed_data[key])))

        json.dump(
            processed_data[key],
            open(
                os.path.join(processed_data_dir, "slot_{}_data.json".format(key)),
                "w",
                encoding="utf-8",
            ),
            indent=2,
            ensure_ascii=False,
        )

    print("sentence label num:", len(all_intent))
    print("tag num:", len(all_tag))
    json.dump(
        all_intent,
        open(
            os.path.join(processed_data_dir, "intent_vocab.json"), "w", encoding="utf-8"
        ),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        all_tag,
        open(os.path.join(processed_data_dir, "tag_vocab.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mode = "all"
    preprocess(mode)
