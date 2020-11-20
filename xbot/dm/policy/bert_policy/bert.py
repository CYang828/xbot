import os
import json

import torch

from transformers import BertForSequenceClassification
from pytorch_lightning import LightningModule


class BertForDialoguePolicy(LightningModule):

    def __init__(self, config):
        super(BertForDialoguePolicy, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(config['pytorch_model'])

    def forward(self, input_ids, attention_mask, token_type_ids):
        pass
