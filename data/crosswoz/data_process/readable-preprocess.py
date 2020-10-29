#!/usr/bin/env python3
import json
import os
import zipfile
import sys
from collections import Counter, OrderedDict
from transformers import BertTokenizer


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def preprocess(mode):
    assert mode == 'all' or mode == 'usr' or mode == 'sys'

    # path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '..')
    processed_data_dir = os.path.join(cur_dir, '../../../xbot/data/{}_data'.format(mode))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    data_key = ['train', 'val', 'test']
    data = {}
    for key in data_key:
        # read crosswoz source data from json.zip
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    all_intent = []
    all_tag = []

    # generate train, val, test dataset
    for key in data_key:
        sessions = []
        for no, sess in data[key].items():
            processed_data = OrderedDict()
            processed_data['sys-usr'] = sess['sys-usr']
            processed_data['type'] = sess['type']
            processed_data['task description'] = sess['task description']
            messages = sess['messages']
            processed_data['turns'] = [OrderedDict({'role': message['role'],
                                                    'utterance': message['content'],
                                                    'dialog_act': message['dialog_act']})
                                       for message in messages]
            sessions.append(processed_data)
        json.dump(sessions,
                  open(os.path.join(processed_data_dir, f'readabe_{key}_data.json'), 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False, sort_keys=False)
    #     all_intent = [x[0] for x in dict(Counter(all_intent)).items()]
    #     all_tag = [x[0] for x in dict(Counter(all_tag)).items()]
    #     print('loaded {}, size {}'.format(key, len(processed_data[key])))
    #     json.dump(processed_data[key],
    #               open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w', encoding='utf-8'),
    #               indent=2, ensure_ascii=False)
    #
    # print('sentence label num:', len(all_intent))
    # print('tag num:', len(all_tag))
    # print(all_intent)
    # json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w', encoding='utf-8'), indent=2,
    #           ensure_ascii=False)
    # json.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.json'), 'w', encoding='utf-8'), indent=2,
    #           ensure_ascii=False)


if __name__ == '__main__':
    # dialogue role: all, usr, sys
    preprocess(sys.argv[1])
