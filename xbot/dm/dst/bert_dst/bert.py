import os
import json
from functools import partial

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, BertTokenizer

from xbot.util.dst_util import DST
from xbot.util.state import default_state
from xbot.util.download import download_from_url
from xbot.util.path import get_data_path, get_root_path, get_config_path
from data.crosswoz.data_process.dst.bert_preprocess import turn2examples, DSTDataset, collate_fn


class BertDST(DST):
    infer_config_name = 'dst/bert/inference.json'
    common_config_name = 'dst/bert/common.json'

    data_urls = {
        'cleaned_ontology.json': '',
        'config.json': '',
        'pytorch_model.bin': '',
        'vocab.txt': ''
    }

    def __init__(self):
        super(BertDST, self).__init__()
        # load config
        root_path = get_root_path()
        common_config_path = os.path.join(get_config_path(), BertDST.common_config_name)
        infer_config_path = os.path.join(get_config_path(), BertDST.infer_config_name)
        common_config = json.load(open(common_config_path))
        infer_config = json.load(open(infer_config_path))
        infer_config.update(common_config)
        infer_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        infer_config['data_path'] = os.path.join(get_data_path(), 'crosswoz/dst_bert_data')
        infer_config['output_dir'] = os.path.join(root_path, infer_config['output_dir'])
        if not os.path.exists(infer_config['data_path']):
            os.makedirs(infer_config['data_path'])
        if not os.path.exists(infer_config['output_dir']):
            os.makedirs(infer_config['output_dir'])

        # download data
        for data_key, url in BertDST.data_urls.items():
            if 'ontology' in data_key:
                dst = os.path.join(infer_config['data_path'], data_key)
            else:
                model_dir = os.path.join(infer_config['data_path'], 'trained_model')
                if not os.path.exists(model_dir):
                    infer_config['model_dir'] = model_dir
                    os.makedirs(model_dir)
                dst = os.path.join(model_dir, data_key)
            file_name = data_key.split('.')[0]
            infer_config[file_name] = dst
            if not os.path.exists(dst):
                download_from_url(url, dst)

        self.ontology = json.load(open(infer_config['ontology'], 'r', encoding='utf8'))
        self.model = BertForSequenceClassification.from_pretrained(infer_config['model_dir'])
        self.tokenizer = BertTokenizer.from_pretrained(infer_config['model_dir'])
        self.config = infer_config

        self.model.eval()
        self.state = default_state()
        self.domains = set(self.state['belief_state'].keys())

    def preprocess(self, sys_uttr, usr_uttr):
        context = sys_uttr + self.tokenizer.sep_token + usr_uttr
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)

        examples = []
        for domain_slots, values in self.ontology.items():
            domain, slot = domain_slots.split('-')

            example = turn2examples(self.tokenizer, domain, 'Request', slot, context_ids)
            examples.append(example)

            for value in values:
                value = ''.join(value.split(' '))
                example = turn2examples(self.tokenizer, domain, slot, value, context_ids)
                examples.append(example)

        examples = list(zip(*examples))
        dataset = DSTDataset(examples)
        collate = partial(collate_fn, mode='infer')
        dataloader = DataLoader(dataset=dataset, batch_size=self.config['batch_size'], collate_fn=collate,
                                shuffle=False, num_workers=self.config['num_workers'])
        return dataloader

    def init_session(self):
        self.state = default_state()

    def update(self, action):
        usr_utter = self.state['history'][-1][1]
        sys_uttr = ''
        if len(self.state['history']) > 1:
            sys_uttr = self.state['history'][-2][1]

        # forward
        dataloader = self.preprocess(sys_uttr, usr_utter)
        pbar = tqdm(enumerate(dataloader), desc='Inferring')
        with torch.no_grad():
            for step, batch in pbar:
                inputs = {k: v.to(self.config['device']) for k, v in list(batch.items())}
                logits = self.model(**inputs)

                preds = logits.argmax(dim=-1).cpu().tolist()

                pred_labels = []
                for i, pred in enumerate(preds):
                    triple = (batch['domains'][i], batch['slots'][i], batch['values'][i])
                    if pred == 1:
                        pred_labels.append(triple)

        # update
        for domain, slot, value in pred_labels:
            if slot == 'Request':
                self.state['request_slots'].append([domain, value])
            else:
                if domain not in self.domains:
                    continue
                if slot in self.state['belief_state'][domain]:
                    self.state['belief_state'][domain][slot] = value


if __name__ == '__main__':
    pass
