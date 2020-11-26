import os
from tqdm import tqdm
from collections import Counter, defaultdict

from xbot.util.path import get_data_path
from xbot.util.file_util import read_zipped_json, dump_json, load_json

# DOMAINS = load_json(os.path.join(get_data_path(), 'crosswoz/policy_bert_data/domains.json'))
# INTENTS = load_json(os.path.join(get_data_path(), 'crosswoz/policy_bert_data/intents.json'))
# SLOTS = load_json(os.path.join(get_data_path(), 'crosswoz/policy_bert_data/slots.json'))
#
# DIS = DOMAINS + INTENTS + SLOTS
# DIS_LEN = len(DIS)

try:
    ACT_ONTOLOGY = load_json(os.path.join(get_data_path(), 'crosswoz/policy_bert_data/act_ontology.json'))
    NUM_ACT = len(ACT_ONTOLOGY)
except FileNotFoundError:
    def preprocess(raw_data_path, output_path, data_type):
        raw_data = read_zipped_json(raw_data_path, data_type)

        act_ontology = set()
        for dial_id, dial in tqdm(raw_data.items(), desc=f'Preprocessing {data_type}'):
            for turn_id, turn in enumerate(dial['messages']):
                if turn['role'] == 'sys' and turn['dialog_act']:
                    for da in turn['dialog_act']:
                        act = '-'.join([da[1], da[0], da[2]])
                        act_ontology.add(act)

        dump_json(list(act_ontology), output_path)


    data_path = get_data_path()
    raw_data_path = os.path.join(data_path, 'crosswoz/raw/train.json.zip')
    output_path = os.path.join(data_path, 'crosswoz/policy_bert_data/act_ontology.json')
    preprocess(raw_data_path, output_path, 'train.json')


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

                cur_domain, act_vecs = get_label_and_domain(turn)
                source = get_source(cur_domain, turn)

                example = {
                    'dial_id': dial_id,
                    'turn_id': turn_id,
                    'source': source,
                    'sys_utter': sys_utter,
                    'usr_utter': usr_utter,
                    'label': act_vecs
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
    main_domains = {"酒店", "地铁", "景点", "餐馆", "出租"}
    domain_counter = defaultdict(int)
    act_vecs = [0 for _ in range(NUM_ACT)]
    cur_domain = None

    for da in turn['dialog_act']:
        intent, domain, slot, value = da
        act = '-'.join([domain, intent, slot])
        # TODO 重新训练，假如当 greet 和 酒店统计频率一致，可能
        #  选到 greet，加上这个判断重新生成数据集、训练
        if domain in main_domains:
            domain_counter[domain] += 1
        act_vecs[ACT_ONTOLOGY.index(act)] = 1

    if domain_counter:
        cur_domain = Counter(domain_counter).most_common(1)[0][0]
    return cur_domain, act_vecs


def eval_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).cpu().sum().item()
    fp = ((preds == 1) & (labels == 0)).cpu().sum().item()
    fn = ((preds == 0) & (labels == 1)).cpu().sum().item()
    tn = ((preds == 0) & (labels == 0)).cpu().sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.
    recall = tp / (tp + fn) if (tp + fn) else 0.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'acc': acc
    }


def get_joint_acc(results):
    joint_acc = []
    for _, dial in results.items():
        for _, turn in dial.items():
            if turn['preds'] == turn['labels']:
                joint_acc.append(1)
            else:
                joint_acc.append(0)

    joint_acc = sum(joint_acc) / len(joint_acc)
    return joint_acc
