import os
import json

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset

from xbot.util.path import get_data_path
from xbot.util.file_util import read_zipped_json

MAX_SEQ_LEN = 512


class DSTDataset(Dataset):

    def __init__(self, examples):
        super(DSTDataset, self).__init__()
        (self.input_ids, self.token_type_ids, self.labels, self.dialogue_idxs,
         self.turn_ids, self.domains, self.slots, self.values, self.belief_states) = examples

    def __getitem__(self, index):
        return (self.input_ids[index], self.token_type_ids[index], self.labels[index], self.dialogue_idxs[index],
                self.turn_ids[index], self.domains[index], self.slots[index], self.values[index],
                self.belief_states[index])

    def __len__(self):
        return len(self.labels)


def collate_fn(examples, mode='train'):
    batch_examples = {}
    examples = list(zip(*examples))
    (batch_examples['labels'], batch_examples['dialogue_idxs'], batch_examples['turn_ids'], batch_examples['domains'],
     batch_examples['slots'], batch_examples['values'], batch_examples['belief_states']) = examples[2:]

    input_ids = examples[0]
    token_type_ids = examples[1]
    max_seq_len = min(max(len(input_id) for input_id in input_ids), MAX_SEQ_LEN)
    input_ids_tensor = torch.zeros((len(input_ids), max_seq_len), dtype=torch.long)
    token_type_ids_tensor = torch.zeros_like(input_ids_tensor)
    attention_mask = torch.ones_like(input_ids_tensor)

    for i, input_id in enumerate(input_ids):
        cur_seq_len = len(input_id)
        if cur_seq_len <= max_seq_len:
            input_ids_tensor[i, :cur_seq_len] = torch.tensor(input_id, dtype=torch.long)
            token_type_ids_tensor[i, :cur_seq_len] = torch.tensor(token_type_ids[i], dtype=torch.long)
            attention_mask[i, cur_seq_len:] = 0
        else:
            input_ids_tensor[i] = torch.tensor(input_id[:max_seq_len - 1] + [102], dtype=torch.long)
            token_type_ids_tensor[i] = torch.tensor(token_type_ids[i][:max_seq_len], dtype=torch.long)

    data = {
        'input_ids': input_ids_tensor,
        'token_type_ids': token_type_ids_tensor,
        'attention_mask': attention_mask,
        'labels': torch.tensor(batch_examples['labels'], dtype=torch.long),
    }

    if mode != 'train':
        data.update({
            'dialogue_idxs': batch_examples['dialogue_idxs'],
            'turn_ids': batch_examples['turn_ids'],
            'domains': batch_examples['domains'],
            'slots': batch_examples['slots'],
            'values': batch_examples['values'],
            'belief_states': batch_examples['belief_states']
        })

    return data


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


if __name__ == '__main__':
    for data_type in ['train', 'dev', 'test']:
        merge_raw_date(data_type)
