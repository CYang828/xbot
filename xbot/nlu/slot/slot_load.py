# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 7:46 下午
# @Author  : zhengjiawei
# @FileName: slot_load.py
# @Software: PyCharm

import os
import zipfile
import json
import torch
import sys
from xbot.util.file_util import cached_path
from nlu import NLU
from data.crosswoz.data_process.nlu_slot_dataloader import Dataloader
from xbot.nlu.slot.slot_bert_model import JointBERT
from data.crosswoz.data_process.nlu_slot_postprocess import  recover_intent
from data.crosswoz.data_process.nlu_slot_preprocess import  preprocess



class BERTSLOT(NLU):
    def __init__(self, mode='all', config_file='crosswoz_all_context_nlu_slot.json',
                 model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip'):
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        curPath = os.path.abspath(os.path.dirname(__file__))
        rootPath = os.path.dirname(os.path.dirname(os.path.dirname(curPath)))
        sys.path.append(rootPath)
        config_path = os.path.join(rootPath, 'xbot/configs/{}'.format(config_file))
        config = json.load(open(config_path))
        DEVICE = config['DEVICE']
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = config['data_dir']
        output_dir = config['output_dir']
        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)
        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json'),encoding="utf-8"))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json'),encoding="utf-8"))
        dataloader = Dataloader(tag_vocab=tag_vocab,intent_vocab=intent_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('tag num:', len(tag_vocab))

        best_model_path = os.path.join(output_dir, 'pytorch_model_nlu_slot.pt')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        print('Load from', best_model_path)

        model = JointBERT(config['model'], DEVICE, dataloader.tag_dim)
        try:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model_nlu_slot.pt'), map_location='cpu'))
        except Exception as e:
            print(e)
        model.to(DEVICE)

        self.model = model
        self.dataloader = dataloader
        print("BERTSLOT loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        ori_tag_seq = ['O'] * len(ori_word_seq)
        context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
        da = {}
        word_seq, tag_seq, new2ori = ori_word_seq, ori_tag_seq, None
        batch_data = [[ori_word_seq, ori_tag_seq,da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to('cpu') for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        slot_logits, batch_slot_loss = self.model(word_seq_tensor,
                                             word_mask_tensor,
                                             tag_seq_tensor,
                                             tag_mask_tensor,
                                             context_seq_tensor,
                                             context_mask_tensor)



        predicts = recover_intent(self.dataloader, slot_logits[0], tag_mask_tensor[0],
                                  batch_data[0][0], batch_data[0][1])
        return predicts



if __name__ == '__main__':
    slot = BERTSLOT(mode='all', config_file='crosswoz_all_context_nlu_slot.json',
                  model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip')
    print(slot.predict(utterance = "北京布提克精品酒店酒店是什么类型，有健身房吗？",context= ['你好，给我推荐一个评分是5分，价格在100-200元的酒店。', '推荐您去北京布提克精品酒店。']))
