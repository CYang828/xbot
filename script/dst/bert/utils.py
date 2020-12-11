import os
import json
from collections import defaultdict
from typing import List, Tuple, Set, Dict

from tqdm import tqdm

import numpy as np

from xbot.util.path import get_data_path
from xbot.util.file_util import read_zipped_json
from data.crosswoz.data_process.dst.bert_preprocess import rank_values


def get_inf_req(triple_list: List[tuple]) -> Tuple[Set[tuple], Set[tuple]]:
    """Save request type and inform type results respectively according to slot type.

    Args:
        triple_list: preds or ground truth (domain, slot, value)

    Returns:
        request, inform triple results
    """
    request = set()
    inform = set()
    for triple in triple_list:
        slot = triple[1]
        if slot == "Request":
            request.add(triple)
        else:
            inform.add(triple)
    return request, inform


def eval_metrics(
    model_output: Dict[str, dict], data_path: str, top_k: int
) -> Dict[str, float]:
    """Calculate `turn_inform` accuracy, `turn_request` accuracy and `joint_goal` accuracy

    Args:
        data_path: save path of bad cases
        model_output: reformatted results containing preds and ground truth
                      according to dialogue id and turn id
        top_k: take top k prediction labels

    Returns:
        metrics
    """
    inform = []
    request = []
    joint_goal = []

    inform_request_dict = {}
    for dialogue_idx, dia in model_output.items():
        turn_dict = defaultdict(dict)
        for turn_id, turn in dia.items():
            logits = turn["logits"]
            preds = turn["preds"]
            labels = turn["labels"]

            preds = rank_values(logits, preds, top_k)

            gold_request, gold_inform = get_inf_req(labels)
            pred_request, pred_inform = get_inf_req(preds)

            turn_dict[turn_id]["pred_inform"] = [list(dsv) for dsv in pred_inform]
            turn_dict[turn_id]["gold_inform"] = [list(dsv) for dsv in gold_inform]
            turn_dict[turn_id]["pred_request"] = [list(dsv) for dsv in pred_request]
            turn_dict[turn_id]["gold_request"] = [list(dsv) for dsv in gold_request]
            request.append(gold_request == pred_request)
            inform.append(gold_inform == pred_inform)

            # 只留下 inform intent，去掉 general intent
            pred_recovered = set(
                [(d, s, v) for d, s, v in pred_inform if not s == v == "none"]
            )
            gold_recovered = set(turn["belief_state"])
            joint_goal.append(pred_recovered == gold_recovered)

        inform_request_dict.update({dialogue_idx: turn_dict})

    with open(os.path.join(data_path, "bad_cases.json"), "w", encoding="utf8") as f:
        json.dump(inform_request_dict, f, indent=2, ensure_ascii=False)

    return {
        "turn_inform": round(float(np.mean(inform)), 3),
        "turn_request": round(float(np.mean(request)), 3),
        "joint_goal": round(float(np.mean(joint_goal)), 3),
    }


def get_recall(data_path):
    with open(os.path.join(data_path, "bad_cases.json"), "r", encoding="utf8") as f:
        bad_cases = json.load(f)
    tp = 0
    total = 0
    for dial_id, dial in bad_cases.items():
        for turn_id, turn in dial.items():
            pred_inform = [tuple(item) for item in turn["pred_inform"]]
            pred_request = [tuple(item) for item in turn["pred_request"]]
            gold_inform = [tuple(item) for item in turn["gold_inform"]]
            gold_request = [tuple(item) for item in turn["gold_request"]]
            tp += len(set(pred_inform) & set(gold_inform))
            tp += len(set(pred_request) & set(gold_request))
            total += len(gold_inform)
            total += len(gold_request)
    print(f"recall: {tp / total}")


def merge_raw_date(data_type: str) -> None:
    """Merge belief state data into user turn

    Args:
        data_type: train, dev or tests
    """
    data_path = get_data_path()
    output_dir = os.path.join(data_path, "crosswoz/dst_bert_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dials_path = os.path.join(
        data_path, "crosswoz/dst_trade_data", f"{data_type}_dials.json"
    )
    raw_filename = "val" if data_type == "dev" else data_type
    raw_path = os.path.join(data_path, "crosswoz/raw", f"{raw_filename}.json.zip")
    dials = json.load(open(dials_path, "r", encoding="utf8"))
    raw = read_zipped_json(raw_path, f"{raw_filename}.json")

    merge_data = {}
    for dial in tqdm(dials, desc=f"Merging {data_type}"):
        dialogue_idx = dial["dialogue_idx"]
        cur_raw = raw[dialogue_idx]
        merge_data[dialogue_idx] = cur_raw
        for turn_id, turn in enumerate(dial["dialogue"]):
            assert merge_data[dialogue_idx]["messages"][2 * turn_id]["role"] == "usr"
            merge_data[dialogue_idx]["messages"][2 * turn_id]["belief_state"] = turn[
                "belief_state"
            ]

    with open(
        os.path.join(output_dir, f"{data_type}4bert_dst.json"), "w", encoding="utf8"
    ) as f:
        json.dump(merge_data, f, ensure_ascii=False, indent=2)


def clean_ontology() -> None:
    """Clean ontology data."""
    data_path = get_data_path()
    ontology_path = os.path.join(data_path, "crosswoz/dst_bert_data/ontology.json")
    ontology = json.load(open(ontology_path, "r", encoding="utf8"))
    cleaned_ontologies = {}
    facility = []
    seps = ["、", "，", ",", ";", "或", "；", "   "]
    for ds, values in tqdm(ontology.items()):
        if len(ds.split("-")) > 2:
            facility.append(ds.split("-")[-1])
            continue
        if ds.split("-")[-1] == "酒店设施":
            continue
        cleaned_values = set()
        for value in values:
            multi_values = [value]
            for sep in seps:
                if sep in value:
                    multi_values = value.split(sep)
                    break
            for v in multi_values:
                v = "".join(v.split())
                cleaned_values.add(v)

        cleaned_ontologies["酒店-酒店设施"] = facility
        cleaned_ontologies[ds] = list(cleaned_values)

    cleaned_ontologies_path = os.path.join(
        data_path, "crosswoz/dst_bert_data/cleaned_ontology.json"
    )
    with open(cleaned_ontologies_path, "w", encoding="utf8") as f:
        json.dump(cleaned_ontologies, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # for data_type in ['train', 'dev', 'tests']:
    #     merge_raw_date(data_type)
    clean_ontology()
