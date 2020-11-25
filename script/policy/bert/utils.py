import os
from tqdm import tqdm
from collections import Counter, defaultdict

from xbot.util.path import get_data_path
from xbot.util.file_util import read_zipped_json, dump_json, load_json

DOMAINS = load_json(os.path.join(get_data_path(), 'crosswoz/policy_bert_data/domains.json'))
INTENTS = load_json(os.path.join(get_data_path(), 'crosswoz/policy_bert_data/intents.json'))
SLOTS = load_json(os.path.join(get_data_path(), 'crosswoz/policy_bert_data/slots.json'))

DIS = DOMAINS + INTENTS + SLOTS
DIS_LEN = len(DIS)


def preprocess(raw_data_path, output_path, data_type):
    raw_data = read_zipped_json(raw_data_path, data_type)

    examples = []
    for dial_id, dial in tqdm(raw_data.items(), desc=f'Preprocessing {data_type}'):
        sys_utter = '对话开始'
        usr_utter = '对话开始'
        for turn_id, turn in enumerate(dial['messages']):
            if turn['role'] == 'usr':
                usr_utter = turn['content']
            elif turn['dialog_act']:

                cur_domain, hierarchical_act_vecs = get_label_and_domain(turn)
                source = get_source(cur_domain, turn)

                example = {
                    'dial_id': dial_id,
                    'turn_id': turn_id,
                    'source': source,
                    'sys_utter': sys_utter,
                    'usr_utter': usr_utter,
                    'label': hierarchical_act_vecs
                }

                examples.append(example)

                sys_utter = turn['content']

    cache_data(data_type, examples, output_path)
    return examples


def cache_data(data_type, examples, output_path):
    dump_json(examples, output_path)
    print(f'Saving preprocessed {data_type} into {output_path} ...')


def get_source(cur_domain, turn):
    db_res = turn['sys_state'].get(cur_domain, None)
    if db_res is None or not db_res['selectedResults']:
        source = '无结果'
    else:
        source = []
        for slot, value in db_res.items():
            if not value or slot == 'selectedResults':
                continue
            source.append(slot + '是' + value)
        source = '，'.join(source)
    return source


def get_label_and_domain(turn):
    domain_counter = defaultdict(int)
    hierarchical_act_vecs = [0 for _ in range(DIS_LEN)]

    for da in turn['dialog_act']:
        intent, domain, slot, value = da
        if '-' in slot:
            slot, value = slot.split('-')
        domain_counter[domain] += 1
        hierarchical_act_vecs[DOMAINS.index(domain)] = 1
        hierarchical_act_vecs[len(DOMAINS) + INTENTS.index(intent)] = 1
        hierarchical_act_vecs[len(DOMAINS) + len(INTENTS) + SLOTS.index(slot)] = 1
    cur_domain = Counter(domain_counter).most_common(1)[0][0]
    return cur_domain, hierarchical_act_vecs
