import os
import json
from typing import Any

from xbot.util.nlu_util import NLU
from xbot.constants import DEFAULT_MODEL_PATH
from xbot.util.path import get_root_path, get_config_path, get_data_path
from xbot.util.download import download_from_url
from data.crosswoz.data_process.nlu_intent_dataloader import Dataloader
from data.crosswoz.data_process.nlu_intent_postprocess import recover_intent

import torch
from torch import nn
from transformers import BertModel


# def recover_intent_predict(dataloader, intent_logits):
#     das = []
#
#     max_index = torch.argsort(intent_logits, descending=True).numpy()
#     for j in max_index[0:5]:
#         intent, domain, slot, value = re.split(r'\+', dataloader.id2intent[j])
#         das.append([intent, domain, slot, value])
#     return das


class IntentWithBert(nn.Module):
    """Intent Classification with Bert"""

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, model_config, device, intent_dim, intent_weight=None):
        super(IntentWithBert, self).__init__()
        # count of intent
        self.intent_num_labels = intent_dim
        # init intent weight
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.] * intent_dim)
        # gpu
        self.device = device

        # load pretrain model from model hub
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        self.dropout = nn.Dropout(model_config['dropout'])
        self.finetune = model_config['finetune']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:
            self.intent_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
            self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
        else:
            self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)

        # Binary Cross Entropy
        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)

    def forward(self, word_seq_tensor, word_mask_tensor, intent_tensor=None):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():  # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:  # 更新参数
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)
        pooled_output = outputs[1]

        if self.hidden_units > 0:
            pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        intent_loss = None
        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
        return intent_logits, intent_loss


class IntentWithBertPredictor(NLU):
    """NLU Intent Classification with Bert 预测器"""

    default_model_config = 'nlu/crosswoz_all_context_nlu_intent.json'
    default_model_name = 'pytorch-intent-with-bert_policy.pt'
    default_model_url = 'http://qiw2jpwfc.hn-bkt.clouddn.com/pytorch-intent-with-bert.pt'

    def __init__(self):
        # path
        root_path = get_root_path()

        config_file = os.path.join(get_config_path(), IntentWithBertPredictor.default_model_config)

        # load config
        config = json.load(open(config_file))
        self.device = config['DEVICE']

        # load intent vocabulary and dataloader
        intent_vocab = json.load(open(os.path.join(get_data_path(),
                                                   'crosswoz/nlu_intent_data/intent_vocab.json'),
                                      encoding='utf-8'))
        dataloader = Dataloader(intent_vocab=intent_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])
        # load best model
        best_model_path = os.path.join(os.path.join(root_path, DEFAULT_MODEL_PATH),
                                       IntentWithBertPredictor.default_model_name)
        # best_model_path = os.path.join(DEFAULT_MODEL_PATH, IntentWithBertPredictor.default_model_name)
        if not os.path.exists(best_model_path):
            download_from_url(IntentWithBertPredictor.default_model_url,
                              best_model_path)
        model = IntentWithBert(config['model'], self.device, dataloader.intent_dim)
        model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        model.to(self.device)
        model.eval()
        self.model = model
        self.dataloader = dataloader
        print(f"{best_model_path} loaded - {best_model_path}")

    def predict(self, utterance, context=list()):
        # utterance
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        # tag
        # ori_tag_seq = ['O'] * len(ori_word_seq)
        intents = []

        word_seq, new2ori = ori_word_seq, None
        batch_data = [[ori_word_seq, intents, word_seq, self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.device) for t in pad_batch)
        word_seq_tensor, word_mask_tensor, intent_tensor = pad_batch
        # inference
        intent_logits,_ = self.model(word_seq_tensor, word_mask_tensor)
        # postprocess
        intent = recover_intent(self.dataloader, intent_logits[0])
        return intent


if __name__ == '__main__':
    nlu = IntentWithBertPredictor()
    print(nlu.predict("北京布提克精品酒店酒店是什么类型，有健身房吗？"))
