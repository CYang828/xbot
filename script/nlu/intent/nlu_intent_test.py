import os
import json
import random

from xbot.constants import DEFAULT_MODEL_PATH
from xbot.util.path import get_root_path, get_config_path, get_data_path
from xbot.nlu.intent.intent_with_bert import IntentWithBert, IntentWithBertPredictor
from xbot.util.download import download_from_url
from data.crosswoz.data_process.nlu_intent_dataloader import Dataloader
from data.crosswoz.data_process.nlu_intent_postprocess import (
    recover_intent,
    calculate_f1,
)

import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    data_urls = {
        "intent_train_data.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/intent_train_data.json",
        "intent_val_data.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/intent_val_data.json",
        "intent_test_data.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/intent_test_data.json",
    }

    # path
    root_path = get_root_path()
    config_file = os.path.join(
        get_config_path(), IntentWithBertPredictor.default_model_config
    )
    config = json.load(open(config_file))
    data_dir = os.path.join(get_data_path(), "crosswoz/nlu_intent_data/")
    output_dir = config["output_dir"]
    output_dir = os.path.join(root_path, output_dir)
    log_dir = config["log_dir"]
    log_dir = os.path.join(root_path, log_dir)
    device = config["DEVICE"]

    # download data
    for data_key, url in data_urls.items():
        dst = os.path.join(os.path.join(data_dir, data_key))
        if not os.path.exists(dst):
            download_from_url(url, dst)

    set_seed(config["seed"])

    intent_vocab = json.load(
        open(os.path.join(data_dir, "intent_vocab.json"), encoding="utf-8")
    )
    dataloader = Dataloader(
        intent_vocab=intent_vocab,
        pretrained_weights=config["model"]["pretrained_weights"],
    )
    for data_key in ["val", "tests"]:
        dataloader.load_data(
            json.load(
                open(
                    os.path.join(data_dir, "intent_{}_data.json".format(data_key)),
                    encoding="utf-8",
                )
            ),
            data_key,
            cut_sen_len=0,
            use_bert_tokenizer=config["use_bert_tokenizer"],
        )
        print("{} set size: {}".format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # load best model
    best_model_path = os.path.join(
        os.path.join(root_path, DEFAULT_MODEL_PATH),
        IntentWithBertPredictor.default_model_name,
    )
    if not os.path.exists(best_model_path):
        download_from_url(IntentWithBertPredictor.default_model_url, best_model_path)

    # model
    model = IntentWithBert(config["model"], device, dataloader.intent_dim)
    model.load_state_dict(torch.load(best_model_path, device))
    model.to(device)
    model.eval()

    batch_size = config["model"]["batch_size"]
    data_key = "tests"
    predict_golden = {"intent": []}
    intent_loss = 0
    for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(
        batch_size, data_key=data_key
    ):
        pad_batch = tuple(t.to(device) for t in pad_batch)
        word_seq_tensor, word_mask_tensor, intent_tensor = pad_batch

        with torch.no_grad():
            intent_logits, batch_intent_loss = model.forward(
                word_seq_tensor, word_mask_tensor, intent_tensor
            )

        intent_loss += batch_intent_loss.item() * real_batch_size
        for j in range(real_batch_size):
            predicts = recover_intent(dataloader, intent_logits[j])
            labels = ori_batch[j][1]

            predict_golden["intent"].append(
                {"predict": [x for x in predicts], "golden": [x for x in labels]}
            )

    total = len(dataloader.data[data_key])
    intent_loss /= total
    print("%d samples %s" % (total, data_key))
    print("\t intent loss:", intent_loss)

    for x in ["intent"]:
        precision, recall, F1 = calculate_f1(predict_golden[x])
        print("-" * 20 + x + "-" * 20)
        print("\t Precision: %.2f" % (precision))
        print("\t Recall: %.2f" % (recall))
        print("\t F1: %.2f" % (F1))

    output_file = os.path.join(output_dir, "output.json")
    json.dump(
        predict_golden["intent"],
        open(output_file, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
