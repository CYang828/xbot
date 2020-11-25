import os
import json

import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy, Precision, Recall, Fbeta

from transformers import BertForSequenceClassification, BertConfig, AdamW

from xbot.util.policy_util import Policy
from script.policy.bert.utils import DIS_LEN
from xbot.util.download import download_from_url
from xbot.util.path import get_data_path, get_config_path, get_root_path


class BertForDialoguePolicyModel(LightningModule):

    def __init__(self, config):
        super(BertForDialoguePolicyModel, self).__init__()
        self.config = config
        model_config = BertConfig.from_pretrained(config['config'])
        model_config.num_labels = DIS_LEN
        self.bert = BertForSequenceClassification.from_pretrained(config['pytorch_model'],
                                                                  config=model_config)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.train_acc = Accuracy(threshold=0.4)
        self.valid_acc = Accuracy(threshold=0.4)
        metric_args = {'num_classes': DIS_LEN, 'threshold': 0.4,
                       'multilabel': True, 'average': 'macro'}
        self.valid_precision = Precision(**metric_args)
        self.valid_recall = Recall(**metric_args)
        self.valid_f1 = Fbeta(**metric_args)

    def forward(self, input_ids, attention_mask, token_type_ids):
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)[0]
        return logits

    def training_step(self, batch, batch_idx):
        loss, probs, labels = self.get_loss(batch)
        return {'loss': loss, 'preds': probs, 'target': labels}

    def training_step_end(self, outputs):
        self.train_acc(outputs['preds'], outputs['target'])
        metric_dict = {'training_loss': outputs['loss'], 'training_acc': self.train_acc}
        self.log_dict(metric_dict, prog_bar=True, on_epoch=True)
        # self.log('training_loss', outputs['loss'], on_epoch=True)
        # self.log('training_acc', self.train_acc, on_epoch=True)

    def get_loss(self, batch):
        inputs = {k: v.to(self.config['device']) for k, v in list(batch.items())[:4]}
        labels = inputs.pop('labels')
        logits = self.bert(**inputs)[0]
        loss = self.loss_fct(logits, labels)
        probs = torch.sigmoid(logits)
        return loss, probs, labels

    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self.get_loss(batch)
        return {'loss': loss, 'preds': probs, 'target': labels}

    def validation_epoch_end(self, validation_step_outputs):
        preds = []
        target = []
        for outputs in validation_step_outputs:
            preds.append(outputs['preds'])
            target.append(outputs['target'])

        preds = torch.cat(preds, dim=0)
        target = torch.cat(target, dim=0)

        # TODO: https://github.com/PyTorchLightning/pytorch-lightning/issues/4255
        self.valid_acc(preds, target)
        self.valid_precision(preds, target)
        self.valid_recall(preds, target)
        self.valid_f1(preds, target)

        metric_dict = {
            'acc': self.valid_acc,
            'precision': self.valid_precision,
            'recall': self.valid_recall,
            'f1': self.valid_f1
        }
        self.log_dict(metric_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['learning_rate'])
        return optimizer


class BertPolicy(Policy):
    inference_config_path = 'policy/bert/inference.json'
    common_config_path = 'policy/bert/common.json'

    data_urls = {
        'config.json': '',
        'pytorch_model.bin': '',
        'vocab.txt': ''
    }

    def __init__(self):
        super(BertPolicy, self).__init__()
        # load config
        root_path = get_root_path()
        common_config_path = os.path.join(get_config_path(), BertPolicy.common_config_path)
        infer_config_path = os.path.join(get_config_path(), BertPolicy.inference_config_path)
        common_config = json.load(open(common_config_path))
        infer_config = json.load(open(infer_config_path))
        infer_config.update(common_config)
        infer_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        infer_config['data_path'] = os.path.join(get_data_path(), 'crosswoz/policy_bert_data')
        infer_config['output_dir'] = os.path.join(root_path, infer_config['output_dir'])
        if not os.path.exists(infer_config['data_path']):
            os.makedirs(infer_config['data_path'])
        if not os.path.exists(infer_config['output_dir']):
            os.makedirs(infer_config['output_dir'])

        # download data
        for data_key, url in BertPolicy.data_urls.items():
            model_dir = os.path.join(infer_config['data_path'], 'trained_model')
            if not os.path.exists(model_dir):
                infer_config['model_dir'] = model_dir
                os.makedirs(model_dir)
            dst = os.path.join(model_dir, data_key)
            file_name = data_key.split('.')[0]
            infer_config[file_name] = dst
            if not os.path.exists(dst):
                download_from_url(url, dst)

        self.model = BertForDialoguePolicyModel(infer_config)
