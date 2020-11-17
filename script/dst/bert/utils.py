import os
import json

from tqdm import tqdm

import numpy as np

from xbot.util.path import get_data_path
from xbot.util.file_util import read_zipped_json


def get_inf_req(triple_list):
    request = set()
    inform = set()
    for d, s, v in triple_list:
        if s == 'Request':
            request.add((d, s, v))
        else:
            inform.add((d, s, v))
    return request, inform


def eval_metrics(model_output):
    inform = []
    request = []
    joint_goal = []

    eval_results_dict = {}
    for pred, label, belief_state, dialogue_idx, turn_id in zip(*model_output.values()):
        if dialogue_idx not in eval_results_dict:
            eval_results_dict[dialogue_idx] = {}
        if turn_id not in eval_results_dict[dialogue_idx]:
            eval_results_dict[dialogue_idx][turn_id] = {'preds': [], 'labels': [], 'belief_state': belief_state}

        eval_results_dict[dialogue_idx][turn_id]['preds'].append(pred)
        eval_results_dict[dialogue_idx][turn_id]['labels'].append(label)

    for dialogue_idx, dia in eval_results_dict.items():
        for turn_id, turn in dia.items():
            preds = turn['preds']
            labels = turn['labels']

            gold_request, gold_inform = get_inf_req(labels)
            pred_request, pred_inform = get_inf_req(preds)

            request.append(gold_request == pred_request)
            inform.append(gold_inform == pred_inform)

            # 只留下 inform intent，去掉 general intent
            pred_recovered = set([(d, s, v) for d, s, v in pred_inform if not s == v == 'none'])
            gold_recovered = set(turn['belief_state'])
            joint_goal.append(pred_recovered == gold_recovered)

    return {'turn_inform': round(np.mean(inform), 3), 'turn_request': round(np.mean(request), 3),
            'joint_goal': round(np.mean(joint_goal), 3)}


def merge_raw_date(data_type):
    data_path = get_data_path()
    output_dir = os.path.join(data_path, 'crosswoz/dst_bert_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dials_path = os.path.join(data_path, 'crosswoz/dst_trade_data', f'{data_type}_dials.json')
    raw_filename = "val" if data_type == "dev" else data_type
    raw_path = os.path.join(data_path, 'crosswoz/raw', f'{raw_filename}.json.zip')
    dials = json.load(open(dials_path, 'r', encoding='utf8'))
    raw = read_zipped_json(raw_path, f'{raw_filename}.json')

    merge_data = {}
    for dial in tqdm(dials, desc=f'Merging {data_type}'):
        dialogue_idx = dial['dialogue_idx']
        cur_raw = raw[dialogue_idx]
        merge_data[dialogue_idx] = cur_raw
        for turn_id, turn in enumerate(dial['dialogue']):
            assert merge_data[dialogue_idx]['messages'][2 * turn_id]['role'] == 'usr'
            merge_data[dialogue_idx]['messages'][2 * turn_id]['belief_state'] = turn['belief_state']

    with open(os.path.join(output_dir, f'{data_type}4bert_dst.json'), 'w', encoding='utf8') as f:
        json.dump(merge_data, f, ensure_ascii=False, indent=2)


def clean_ontology():
    data_path = get_data_path()
    ontology_path = os.path.join(data_path, 'crosswoz/dst_bert_data/ontology.json')
    ontology = json.load(open(ontology_path, 'r', encoding='utf8'))
    cleaned_ontologies = {}
    facility = []
    seps = ['、', '，', ',', ';', '或', '；', '   ']
    for ds, values in tqdm(ontology.items()):
        if len(ds.split('-')) > 2:
            facility.append(ds.split('-')[-1])
            continue
        if ds.split('-')[-1] == '酒店设施':
            continue
        cleaned_values = set()
        for value in values:
            multi_values = [value]
            for sep in seps:
                if sep in value:
                    multi_values = value.split(sep)
                    break
            for v in multi_values:
                v = ''.join(v.split())
                cleaned_values.add(v)

        cleaned_ontologies['酒店-酒店设施'] = facility
        cleaned_ontologies[ds] = list(cleaned_values)

    cleaned_ontologies_path = os.path.join(data_path, 'crosswoz/dst_bert_data/cleaned_ontology.json')
    with open(cleaned_ontologies_path, 'w', encoding='utf8') as f:
        json.dump(cleaned_ontologies, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # for data_type in ['train', 'dev', 'test']:
    #     merge_raw_date(data_type)
    clean_ontology()
