import json
import os
import zipfile
import sys
from collections import Counter
from transformers import BertTokenizer


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, "r")
    return json.load(archive.open(filename))


def preprocess():
    cur_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在的文件夹路径
    data_dir = os.path.join(cur_dir, "../../../data/crosswoz/raw")
    processed_data_dir = os.path.join(cur_dir, "../nlu_intent_data/")
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    print(data_dir)
    data_key = ["train", "val", "tests"]
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(
            os.path.join(data_dir, key + ".json.zip"), key + ".json"
        )
        print("load {}, size {}".format(key, len(data[key])))

    processed_data = {}  ##存储训练集验证集和测试集处理好的数据
    all_intent = []

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert_policy-wwm-ext")

    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            for i, turn in enumerate(sess["messages"]):
                utterance = turn["content"]
                tokens = tokenizer.tokenize(utterance)

                intents = []
                for intent, domain, slot, value in turn["dialog_act"]:  # 从字段中提取标签
                    intents.append("+".join([intent, domain]))

                processed_data[key].append([tokens, intents])  # 将分词之后的一条数据和标签存放在一起
                all_intent += intents
        all_intent = [
            x[0] for x in dict(Counter(all_intent)).items()
        ]  # {"intent":3}  只取出来字典中的key
        print("loaded {}, size {}".format(key, len(processed_data[key])))
        json.dump(
            processed_data[key],
            open(
                os.path.join(processed_data_dir, "intent_{}_data.json".format(key)),
                "w",
                encoding="utf-8",
            ),
            indent=2,
            ensure_ascii=False,
        )

    print("sentence label num:", len(all_intent))
    print(all_intent)
    json.dump(
        all_intent,
        open(
            os.path.join(processed_data_dir, "intent_vocab.json"), "w", encoding="utf-8"
        ),
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    preprocess()
