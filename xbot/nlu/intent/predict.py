import os
import zipfile
import json

from xbot.util.nlu_util import NLU
from xbot.gl import DEFAULT_MODEL_PATH
from xbot.util.file_util import cached_path
from xbot.nlu.intent.intent_with_bert import IntentWithBert
from data.crosswoz.data_process.nlu_intent_dataloader import Dataloader
from data.crosswoz.data_process.nlu_intent_postprocess import recover_intent
from xbot.util.download import download_from_url

import torch


class BERTNLU(NLU):
    default_model_name = 'pytorch-intent-with-bert.bin'

    def __init__(self, config_file='crosswoz_all_context.json',
                 model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip'):
        # path
        current_path = os.path.abspath(os.path.dirname(__file__))
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
        config_file = os.path.join(root_path, 'xbot/configs/{}'.format(config_file))

        # load config
        config = json.load(open(config_file))
        data_path = os.path.join(root_path, config['data_dir'])
        device = config['DEVICE']

        # load intent vocabulary and dataloader
        intent_vocab = json.load(open(os.path.join(data_path, 'intent_vocab.json'), encoding='utf-8'))
        dataloader = Dataloader(intent_vocab=intent_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])
        # load best model
        best_model_path = os.path.join(DEFAULT_MODEL_PATH, 'pytorch-intent-with-bert.bin')
        if not os.path.exists(best_model_path):
            download_from_url('http://qiw2jpwfc.hn-bkt.clouddn.com/pytorch-intent-with-bert.bin',
                              best_model_path)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_path)
            archive.close()
        print('Load from', best_model_path)
        model = IntentWithBert(config['model'], device, dataloader.intent_dim)
        try:
            model.load_state_dict(torch.load(os.path.join(output_path, 'pytorch-intent-with-bert.bin'),
                                             map_location='cpu'))
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
    nlu = BERTNLU(config_file='crosswoz_all_context.json',
                  model_file='https://convlab.blob.core.windows.net/convlab-2/bert_crosswoz_all_context.zip')
    print(nlu.predict("北京布提克精品酒店酒店是什么类型，有健身房吗？"))
