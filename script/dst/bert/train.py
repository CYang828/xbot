import os
import json
import time
import random
import warnings
from copy import deepcopy
from functools import partial
from collections import defaultdict
from concurrent import futures

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.multiprocessing import Manager, Pool

from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification

from script.dst.bert.utils import eval_metrics
from xbot.util.download import download_from_url
from xbot.util.path import get_data_path, get_root_path, get_config_path
from data.crosswoz.data_process.dst.bert_preprocess import turn2examples, DSTDataset, collate_fn

warnings.simplefilter('ignore')


class Trainer:

    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['vocab'])
        self.ontology = json.load(open(config['cleaned_ontology'], 'r', encoding='utf8'))
        start_time = time.time()
        self.train_dataloader = self.load_data(data_path=self.config['train4bert_dst'], data_type='train')
        self.eval_dataloader = self.load_data(data_path=self.config['dev4bert_dst'], data_type='dev')
        self.test_dataloader = self.load_data(data_path=self.config['test4bert_dst'], data_type='test')
        elapsed = time.time() - start_time
        print(f'Loading data cost {elapsed}s ...')
        self.best_model_path = None
        self.model = None
        self.model_config = None
        self.optimizer = None

    def set_model(self):
        self.model_config = BertConfig.from_pretrained(self.config['config'])
        self.model = BertForSequenceClassification.from_pretrained(self.config['pytorch_model'],
                                                                   config=self.model_config)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

        self.model.to(self.config['device'])
        if self.config['n_gpus'] > 1:
            self.model = nn.DataParallel(self.model)

    @staticmethod
    def get_pos_neg_examples(example, pos_examples, neg_examples):
        if example[-3] == 1:
            pos_examples.append(example)
        else:
            neg_examples.append(example)

    def iter_dials(self, dials, data_type, pos_examples, neg_examples, process_id):
        for dial_id, dial in tqdm(dials, desc=f'Building {data_type} examples, current process-{process_id}'):
            sys_utter = ''
            for turn_id, turn in enumerate(dial['messages']):
                if turn['role'] == 'sys':
                    sys_utter = turn['content']
                else:
                    belief_state = []
                    for bs in turn['belief_state']:
                        domain, slot = bs['slots'][0][0].split('-')
                        value = ''.join(bs['slots'][0][1].split(' '))
                        belief_state.append((domain, slot, value))

                    usr_utter = turn['content']
                    context = sys_utter + self.tokenizer.sep_token + usr_utter
                    context_ids = self.tokenizer.encode(context, add_special_tokens=False)

                    turn_labels = turn['dialog_act']
                    triple_labels = set()
                    for usr_da in turn_labels:
                        intent, domain, slot, value = usr_da
                        if intent == 'Request':
                            triple_labels.add((domain, 'Request', slot))
                        else:
                            if '-' in slot:  # 酒店设施
                                slot, value = slot.split('-')
                            triple_labels.add((domain, slot, value))

                    for domain_slots, values, in self.ontology.items():
                        domain_slot = domain_slots.split('-')
                        domain, slot = domain_slot

                        if domain in ['reqmore']:
                            continue

                        if domain not in ['greet', 'welcome', 'thank', 'bye'] and slot != '酒店设施':
                            example = turn2examples(self.tokenizer, domain, 'Request', slot, context_ids,
                                                    triple_labels, belief_state, dial_id, turn_id)
                            self.get_pos_neg_examples(example, pos_examples, neg_examples)

                        for value in values:
                            value = ''.join(value.split(' '))
                            if slot == '酒店设施':
                                slot_value = slot + f'-{value}'
                                example = turn2examples(self.tokenizer, domain, 'Request', slot_value, context_ids,
                                                        triple_labels, belief_state, dial_id, turn_id)
                                self.get_pos_neg_examples(example, pos_examples, neg_examples)

                            example = turn2examples(self.tokenizer, domain, slot, value, context_ids,
                                                    triple_labels, belief_state, dial_id, turn_id)

                            self.get_pos_neg_examples(example, pos_examples, neg_examples)

    def build_examples(self, data_path, data_cache_path, data_type):
        dials = json.load(open(data_path, 'r', encoding='utf8'))
        dials = list(dials.items())
        if self.config['debug']:
            dials = dials[:self.config['num_processes']]

        num_orig_dials = len(dials)
        num_sampling_dials = int(len(dials) * self.config['overall_undersampling_ratio'])
        dials = random.sample(dials, k=num_sampling_dials)
        print(f'After overall undersampling, {num_orig_dials} dialogues reduce to {num_sampling_dials} ...')

        neg_examples = Manager().list()
        pos_examples = Manager().list()

        dials4single_process = (len(dials) - 1) // self.config['num_processes'] + 1
        print(f'Single process have {dials4single_process} examples ...')
        pool = Pool(self.config['num_processes'])
        for i in range(self.config['num_processes']):
            pool.apply_async(func=self.iter_dials,
                             args=(dials[dials4single_process * i: dials4single_process * (i + 1)],
                                   data_type, pos_examples, neg_examples, i))
        pool.close()
        pool.join()

        pos_examples = list(pos_examples)
        neg_examples = list(neg_examples)

        examples = pos_examples + neg_examples
        print(f'{len(dials)} dialogs generate {len(examples)} examples ...')
        print(f'[neg:pos]: {len(neg_examples) / len(pos_examples)}')

        if self.config['random_undersampling']:
            neg_examples = random.sample(neg_examples, k=len(pos_examples))
            examples = pos_examples + neg_examples
            print(f'After undersampling, remain total {len(examples)} examples')

        random.shuffle(examples)
        examples = list(zip(*examples))
        torch.save(examples, data_cache_path)

        return examples

    def load_data(self, data_path, data_type):
        print(f'Starting preprocess {data_type} data ...')
        raw_data_name = os.path.basename(data_path)
        processed_data_name = 'processed_' + raw_data_name.split('.')[0] + '.pt'
        data_cache_path = os.path.join(self.config['data_path'], processed_data_name)
        if self.config['use_cache_data'] and os.path.exists(data_cache_path):
            print(f'Loading cache {data_type} data ...')
            examples = torch.load(data_cache_path)
            print(f'Total {len(examples)} {data_type} examples ...')
        else:
            examples = self.build_examples(data_path, data_cache_path, data_type)
        dataset = DSTDataset(examples)
        shuffle = True if data_type == 'train' else False
        batch_size = self.config[f'{data_type}_batch_size']
        collate = partial(collate_fn, mode=data_type)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=self.config['num_workers'], collate_fn=collate)
        return dataloader

    def evaluation(self, epoch=None, mode='dev'):
        self.model.eval()
        eval_bar = tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), desc='Evaluating')
        results = defaultdict(list)
        with torch.no_grad():
            for step, batch in eval_bar:
                inputs = {k: v.to(self.config['device']) for k, v in list(batch.items())[:4]}
                loss, logits = self.model(**inputs)[:2]

                if self.config['n_gpus'] > 1:
                    loss = loss.mean()

                preds = logits.argmax(dim=-1).cpu().tolist()
                labels = inputs['labels'].cpu().tolist()

                pred_labels = []
                ground_labels = []
                for i, (pred, label) in enumerate(zip(preds, labels)):
                    triple = (batch['domains'][i], batch['slots'][i], batch['values'][i])
                    if pred == 1:
                        pred_labels.append(triple)
                    if label == 1:
                        ground_labels.append(triple)

                results['pred_labels'].extend(pred_labels)
                results['ground_labels'].extend(ground_labels)
                results['belief_states'].extend(batch['belief_states'])
                results['dialogue_idxs'].extend(batch['dialogue_idxs'])
                results['turn_ids'].extend(batch['turn_ids'])

                desc = f'Evaluating： Epoch: {epoch}, ' if mode == 'dev' else 'Best model, '
                desc += f'CELoss: {loss.item():.3f}'
                eval_bar.set_description(desc)

        metrics_res = eval_metrics(results)
        print('*' * 10 + ' eval metrics ' + '*' * 10)
        print(json.dumps(metrics_res, indent=2))
        return metrics_res[self.config['eval_metric']]

    def eval_test(self):
        if self.best_model_path is not None:
            self.model.module = BertForSequenceClassification.from_pretrained(self.best_model_path)
            self.model.to(self.config['device'])
        self.evaluation(self.test_dataloader, mode='test')

    def train(self):
        self.set_model()
        epoch_bar = trange(0, self.config['num_epochs'], desc='Epoch')
        best_metric = 0

        for epoch in epoch_bar:
            self.model.train()
            train_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc='Training')
            for step, batch in train_bar:
                inputs = {k: v.to(self.config['device']) for k, v in batch.items()}
                loss = self.model(**inputs)[0]

                if self.config['n_gpus'] > 1:
                    loss = loss.mean()

                self.model.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
                self.optimizer.step()

                train_bar.set_description(f'Training： Epoch: {epoch}, Iter: {step}, CELoss: {loss.item():.3f}')

            eval_metric = self.evaluation(epoch)
            if eval_metric > best_metric:
                print(f'Best model saved, {self.config["eval_metric"]}: {eval_metric} ...')
                best_metric = eval_metric
                self.save(epoch, best_metric)

    def save(self, epoch, joint_goal):
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])

        save_name = f'Epoch-{epoch}-JointGoal-{joint_goal:.3f}'
        self.best_model_path = os.path.join(self.config['output_dir'], save_name)
        model_to_save = deepcopy(self.model.module if hasattr(self.model, 'module') else self.model)
        model_to_save.cpu().save_pretrained(self.best_model_path)
        self.tokenizer.save_pretrained(self.best_model_path)


def main():
    model_config_name = 'dst/bert/train.json'
    common_config_name = 'dst/bert/common.json'

    data_urls = {
        'train4bert_dst.json': '',
        'dev4bert_dst.json': '',
        'test4bert_dst.json': '',
        'cleaned_ontology.json': 'http://qiw2jpwfc.hn-bkt.clouddn.com/ontology.json',
        'config.json': '',
        'pytorch_model.bin': '',
        'vocab.txt': ''
    }

    # load config
    root_path = get_root_path()
    common_config_path = os.path.join(get_config_path(), common_config_name)
    train_config_path = os.path.join(get_config_path(), model_config_name)
    common_config = json.load(open(common_config_path))
    train_config = json.load(open(train_config_path))
    train_config.update(common_config)
    train_config['n_gpus'] = torch.cuda.device_count()
    train_config['train_batch_size'] = max(1, train_config['n_gpus']) * train_config['train_batch_size']
    train_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_config['data_path'] = os.path.join(get_data_path(), 'crosswoz/dst_bert_data')
    train_config['output_dir'] = os.path.join(root_path, train_config['output_dir'])
    if not os.path.exists(train_config['data_path']):
        os.makedirs(train_config['data_path'])
    if not os.path.exists(train_config['output_dir']):
        os.makedirs(train_config['output_dir'])

    # download data
    for data_key, url in data_urls.items():
        dst = os.path.join(train_config['data_path'], data_key)
        file_name = data_key.split('.')[0]
        train_config[file_name] = dst
        if not os.path.exists(dst):
            download_from_url(url, dst)

    # train
    trainer = Trainer(train_config)
    trainer.train()
    trainer.eval_test()


if __name__ == '__main__':
    main()
