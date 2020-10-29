import os
import zipfile
import json
import torch
from xbot.util.file_util import cached_path
from xbot.util.nlu_util import NLU
from data.crosswoz.data_process.nlu_intent_dataloader import Dataloader
from xbot.nlu.intent.jointBERT import JointBERT
from data.crosswoz.data_process.nlu_intent_postprocess import recover_intent
import sys
class BERTNLU(NLU):
    def __init__(self, config_file='crosswoz_all_context.json',
                 model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip'):
        curPath = os.path.abspath(os.path.dirname(__file__))
        rootPath = os.path.dirname(os.path.dirname(os.path.dirname(curPath)))
        config_file = os.path.join(rootPath, 'xbot/configs/{}'.format(config_file))

        print(config_file)
        config = json.load(open(config_file))
        DEVICE = config['DEVICE']
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])
        print("data_dir",data_dir)
        print("output_dir",output_dir)

        # if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
        #     preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json'),encoding='utf-8'))
        dataloader = Dataloader(intent_vocab=intent_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))


        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        print('Load from', best_model_path)
        model = JointBERT(config['model'], DEVICE, dataloader.intent_dim)
        try:
            model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location='cpu'))
        except Exception as e:
            print(e)
        model.to("cpu")
        model.eval()

        self.model = model
        self.dataloader = dataloader
        print("BERTNLU loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        ori_tag_seq = ['O'] * len(ori_word_seq)
        intents = []

        batch_data = [[ori_word_seq, ori_tag_seq, intents]]
        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to('cpu') for t in pad_batch)
        word_seq_tensor, intent_tensor, word_mask_tensor = pad_batch
        intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor)

        intent = recover_intent(self.dataloader, intent_logits[0])
        return intent

if __name__ == '__main__':
    nlu = BERTNLU(config_file='crosswoz_all_context.json',model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip')
    print(nlu.predict("北京布提克精品酒店酒店是什么类型，有健身房吗？"))
