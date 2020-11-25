import os
import time
from functools import partial
from collections import defaultdict

from transformers import BertTokenizer

import pytorch_lightning as pl

from torch.utils.data import DataLoader

from xbot.util.path import get_data_path
from xbot.util.train_util import update_config
from xbot.util.file_util import load_json
from script.policy.bert.utils import preprocess
from xbot.util.download import download_from_url
from xbot.dm.policy.bert_policy.bert import BertForDialoguePolicyModel
from data.crosswoz.data_process.policy.bert_proprecess import PolicyDataset, collate_fn


class Trainer:

    def __init__(self, config):
        self.config = config
        self.model = BertForDialoguePolicyModel(config)
        self.tokenizer = BertTokenizer.from_pretrained(config['vocab'])

        start_time = time.time()
        self.train_dataloader = self.load_data('train')
        self.eval_dataloader = self.load_data('val')
        self.test_dataloader = self.load_data('test')
        elapsed_time = time.time() - start_time
        print(f'Loading data cost {elapsed_time}s ...')

    def load_data(self, data_type):
        raw_data_path = os.path.join(self.config['raw_data_path'], f'{data_type}.json.zip')
        filename = f'{data_type}.json'
        output_path = os.path.join(self.config['output_dir'], filename)
        if not self.config['use_data_cache']:
            examples = preprocess(raw_data_path, output_path, filename)
        else:
            print(f'Loading {data_type} data from cache ...')
            examples = load_json(output_path)

        examples_dict = self.get_input_ids(examples)

        print(f'Starting building {data_type} dataset ...')
        dataset = PolicyDataset(**examples_dict)
        shuffle = True if data_type == 'train' else False
        collate = partial(collate_fn, mode=data_type)
        batch_size = self.config[f'{data_type}_batch_size']
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=self.config['num_workers'],
                                shuffle=shuffle, pin_memory=True, collate_fn=collate)
        return dataloader

    def get_input_ids(self, examples):
        examples_dict = defaultdict(list)

        for example in examples:
            sys_utter_tokens = self.tokenizer.tokenize(example['sys_utter'])
            usr_utter_tokens = self.tokenizer.tokenize(example['usr_utter'])
            source_tokens = self.tokenizer.tokenize(example['source'])
            sys_utter_ids = self.tokenizer.convert_tokens_to_ids(sys_utter_tokens)
            usr_utter_ids = self.tokenizer.convert_tokens_to_ids(usr_utter_tokens)
            source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
            input_ids = ([self.tokenizer.cls_token_id] + sys_utter_ids + [self.tokenizer.sep_token_id]
                         + usr_utter_ids + [self.tokenizer.sep_token_id] + source_ids + [self.tokenizer.sep_token_id])
            token_type_ids = ([0] + [0] * (len(sys_utter_ids) + 1) + [1] * (len(usr_utter_ids) + 1)
                              + [0] * (len(source_ids) + 1))

            examples_dict['dial_ids'].append(example['dial_id'])
            examples_dict['turn_ids'].append(example['turn_id'])
            examples_dict['input_ids'].append(input_ids)
            examples_dict['token_type_ids'].append(token_type_ids)
            examples_dict['labels'].append(example['label'])

        return examples_dict

    def train(self):
        pl.seed_everything(self.config['seed'])
        trainer = pl.Trainer(gpus=self.config['n_gpus'],
                             accelerator='dp',
                             max_epochs=self.config['num_epochs'])
        trainer.fit(self.model, train_dataloader=self.train_dataloader,
                    val_dataloaders=self.eval_dataloader)


def main():
    train_config_name = 'policy/bert/train.json'
    common_config_name = 'policy/bert/common.json'

    data_urls = {
        'config.json': 'http://qiw2jpwfc.hn-bkt.clouddn.com/config.json',
        'pytorch_model.bin': 'http://qiw2jpwfc.hn-bkt.clouddn.com/pytorch_model.bin',
        'vocab.txt': 'http://qiw2jpwfc.hn-bkt.clouddn.com/vocab.txt',
        'domains.json': '',
        'intents.json': '',
        'slots.json': ''
    }

    train_config = update_config(common_config_name, train_config_name)
    train_config['raw_data_path'] = os.path.join(get_data_path(), 'crosswoz/raw')

    # download data
    for data_key, url in data_urls.items():
        dst = os.path.join(train_config['data_path'], data_key)
        file_name = data_key.split('.')[0]
        train_config[file_name] = dst
        if not os.path.exists(dst):
            download_from_url(url, dst)

    trainer = Trainer(train_config)
    trainer.train()


if __name__ == '__main__':
    main()
