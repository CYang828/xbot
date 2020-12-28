import os
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Tuple, List, Dict

import torch

from xbot.util.path import get_data_path
from xbot.util.file_util import read_zipped_json, dump_json, load_json


def load_act_ontology() -> Tuple[List[str], int]:
    """Load action ontology from cache.

    Returns:
        action ontology and numbers of action
    """
    act_ontology = load_json(
        os.path.join(get_data_path(), "crosswoz/policy_bert_data/act_ontology.json")
    )
    num_act = len(act_ontology)
    return act_ontology, num_act


try:
    ACT_ONTOLOGY, NUM_ACT = load_act_ontology()
except FileNotFoundError:

    def get_act_ontology(raw_data_path: str, output_path: str) -> None:
        """Generate action ontology from raw train data.

        Args:
            raw_data_path: raw train data path
            output_path: save path of action ontology file
        """
        raw_data = read_zipped_json(raw_data_path, "train.json")

        act_ontology = set()
        for dial_id, dial in tqdm(
            raw_data.items(), desc="Generate action ontology ..."
        ):
            for turn_id, turn in enumerate(dial["messages"]):
                if turn["role"] == "sys" and turn["dialog_act"]:
                    for da in turn["dialog_act"]:
                        act = "-".join([da[1], da[0], da[2]])
                        act_ontology.add(act)

        dump_json(list(act_ontology), output_path)

    data_path = get_data_path()
    raw_data_path = os.path.join(data_path, "crosswoz/raw/train.json.zip")
    output_path = os.path.join(data_path, "crosswoz/policy_bert_data/act_ontology.json")
    get_act_ontology(raw_data_path, output_path)

    ACT_ONTOLOGY, NUM_ACT = load_act_ontology()


def preprocess(
    raw_data_path: str, output_path: str, data_type: str
) -> List[Dict[str, list]]:
    """Preprocess raw data to generate model inputs.

    Args:
        raw_data_path: raw (train, dev, tests) data path
        output_path: save path of precessed data file
        data_type: train, dev or tests

    Returns:
        precessed data
    """
    raw_data = read_zipped_json(raw_data_path, data_type)

    examples = []
    for dial_id, dial in tqdm(raw_data.items(), desc=f"Preprocessing {data_type}"):
        sys_utter = "对话开始"
        usr_utter = "对话开始"
        for turn_id, turn in enumerate(dial["messages"]):
            if turn["role"] == "usr":
                usr_utter = turn["content"]
            elif turn["dialog_act"]:

                cur_domain, act_vecs = get_label_and_domain(turn)
                source = get_source(cur_domain, turn)

                example = {
                    "dial_id": dial_id,
                    "turn_id": turn_id,
                    "source": source,
                    "sys_utter": sys_utter,
                    "usr_utter": usr_utter,
                    "label": act_vecs,
                }

                examples.append(example)

                sys_utter = turn["content"]

    cache_data(data_type, examples, output_path)
    return examples


def cache_data(
    data_type: str, examples: List[Dict[str, list]], output_path: str
) -> None:
    """Save processed data."""
    dump_json(examples, output_path)
    print(f"Saving preprocessed {data_type} into {output_path} ...")


def get_source(cur_domain: str, turn: dict) -> str:
    """Concat slot and value."""
    db_res = turn["sys_state"].get(cur_domain, None)
    if db_res is None or not db_res["selectedResults"]:
        source = "无结果"
    else:
        source = []
        for slot, value in db_res.items():
            if not value or slot == "selectedResults":
                continue
            source.append(slot + "是" + value)
        source = "，".join(source)
    return source


def get_label_and_domain(turn: dict) -> Tuple[str, list]:
    """Construct one-hot label and take current domain."""
    main_domains = {"酒店", "地铁", "景点", "餐馆", "出租"}
    domain_counter = defaultdict(int)
    act_vecs = [0 for _ in range(NUM_ACT)]
    cur_domain = None

    for da in turn["dialog_act"]:
        intent, domain, slot, value = da
        act = "-".join([domain, intent, slot])
        # TODO 重新训练，假如当 greet 和酒店统计频率一致，可能
        #  选到 greet，加上这个判断重新生成数据集、训练
        if domain in main_domains:
            domain_counter[domain] += 1
        act_vecs[ACT_ONTOLOGY.index(act)] = 1

    if domain_counter:
        cur_domain = Counter(domain_counter).most_common(1)[0][0]
    return cur_domain, act_vecs


def eval_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """Calculate metrics: precision, recall, f1, accuracy"""
    tp = ((preds == 1) & (labels == 1)).cpu().sum().item()
    fp = ((preds == 1) & (labels == 0)).cpu().sum().item()
    fn = ((preds == 0) & (labels == 1)).cpu().sum().item()
    tn = ((preds == 0) & (labels == 0)).cpu().sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "acc": acc}


def get_joint_acc(results: Dict[str, dict]) -> float:
    """Calculate joint accuracy.

    Args:
        results: from Trainer.format_results method

    Returns:
        turn joint accuracy
    """
    joint_acc = []
    for _, dial in results.items():
        for _, turn in dial.items():
            if turn["preds"] == turn["labels"]:
                joint_acc.append(1)
            else:
                joint_acc.append(0)

    joint_acc = sum(joint_acc) / len(joint_acc)
    return joint_acc
