import os
import json
import random
import zipfile
from tqdm import tqdm
from copy import deepcopy

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from xbot.dm.dst.rule_dst.rule import RuleDST


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DataPreprocessor:
    def __init__(self, config, vector):
        self.vector = vector
        processed_dir = config["data_path"]
        if os.path.exists(os.path.join(processed_dir, "train.pt")):
            print("Loading processed data file")
            self._load_data(processed_dir)
        else:
            print("Start preprocessing dataset")
            self._build_data(config["raw_data_path"], processed_dir)

    def _build_data(self, root_dir, processed_dir):
        raw_data = {}
        for part in ["train", "val", "tests"]:
            archive = zipfile.ZipFile(os.path.join(root_dir, f"{part}.json.zip"), "r")
            with archive.open(f"{part}.json", "r") as f:
                raw_data[part] = json.load(f)

        self.data = {}
        # for cur domain update
        dst = RuleDST()
        for part in ["train", "val", "tests"]:
            self.data[part] = []

            for key in tqdm(raw_data[part], desc=part, total=len(raw_data[part])):
                sess = raw_data[part][key]["messages"]
                dst.init_session()
                for i, turn in enumerate(sess):
                    if turn["role"] == "usr":
                        dst.update(usr_da=turn["dialog_act"])
                        if i + 2 == len(sess):
                            # 为什么要提前设置呢
                            dst.state["terminated"] = True
                    else:
                        for domain, svs in turn["sys_state"].items():
                            for slot, value in svs.items():
                                if slot != "selectedResults":
                                    # 此处的更新可能会和 user side 的更新重复，sys_state
                                    # 中包含前面用户的 dialogue act 的信息
                                    dst.state["belief_state"][domain][slot] = value
                        action = turn["dialog_act"]
                        # 当前系统的 state 对应的系统的 action
                        self.data[part].append(
                            [
                                self.vector.state_vectorize(deepcopy(dst.state)),
                                self.vector.action_vectorize(action),
                            ]
                        )
                        dst.state["system_action"] = turn["dialog_act"]

        os.makedirs(processed_dir, exist_ok=True)
        for part in ["train", "val", "tests"]:
            torch.save(self.data[part], os.path.join(processed_dir, f"{part}.pt"))

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ["train", "val", "tests"]:
            self.data[part] = torch.load(os.path.join(processed_dir, f"{part}.pt"))

    def create_dataset(self, part, batch_size):
        print(f"Start creating {part} dataset")
        s = []
        a = []
        for item in self.data[part]:
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
        s = torch.stack(s)
        a = torch.stack(a)
        dataset = CrossWozDataset(s, a)
        dataloader = DataLoader(dataset, batch_size, True)
        print(f"Finish creating {part} dataset")
        return dataloader


class CrossWozDataset(Dataset):
    def __init__(self, s_s, a_s):
        self.s_s = s_s
        self.a_s = a_s
        self.num_total = len(s_s)

    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        return s, a

    def __len__(self):
        return self.num_total


def f1(a, target):
    tp, fp, fn = 0, 0, 0
    real = target.nonzero().tolist()
    predict = a.nonzero().tolist()
    for item in real:
        if item in predict:
            tp += 1
        else:
            fn += 1
    for item in predict:
        if item not in real:
            fp += 1
    return tp, fp, fn
