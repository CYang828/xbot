import os
import json
import random
import zipfile

from xbot.util.path import get_root_path, get_config_path, get_data_path
from xbot.util.download import download_from_url
from xbot.nlu.intent.intent_with_bert import IntentWithBert
from data.crosswoz.data_process.nlu_intent_dataloader import Dataloader
from data.crosswoz.data_process.nlu_intent_postprocess import (
    calculate_f1,
    recover_intent,
)

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup


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
    # load config
    root_path = get_root_path()
    config_path = os.path.join(
        os.path.join(get_config_path(), "nlu"), "crosswoz_all_context_nlu_intent.json"
    )
    config = json.load(open(config_path))
    data_path = os.path.join(get_data_path(), "crosswoz/nlu_intent_data/")
    output_dir = config["output_dir"]
    output_dir = os.path.join(root_path, output_dir)
    log_dir = config["log_dir"]
    log_dir = os.path.join(root_path, log_dir)
    device = config["DEVICE"]

    # download data
    for data_key, url in data_urls.items():
        dst = os.path.join(os.path.join(data_path, data_key))
        if not os.path.exists(dst):
            download_from_url(url, dst)

    # seed
    set_seed(config["seed"])

    # load intent vocabulary and dataloader
    intent_vocab = json.load(
        open(os.path.join(data_path, "intent_vocab.json"), encoding="utf-8")
    )
    dataloader = Dataloader(
        intent_vocab=intent_vocab,
        pretrained_weights=config["model"]["pretrained_weights"],
    )

    # load data
    for data_key in ["train", "val", "tests"]:
        dataloader.load_data(
            json.load(
                open(
                    os.path.join(data_path, "intent_{}_data.json".format(data_key)),
                    encoding="utf-8",
                )
            ),
            data_key,
            cut_sen_len=config["cut_sen_len"],
            use_bert_tokenizer=config["use_bert_tokenizer"],
        )
        print("{} set size: {}".format(data_key, len(dataloader.data[data_key])))

    # output and log dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # model
    model = IntentWithBert(
        config["model"], device, dataloader.intent_dim, dataloader.intent_weight
    )
    model.to(device)

    # optimizer and scheduler
    if config["model"]["finetune"]:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": config["model"]["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config["model"]["learning_rate"],
            eps=config["model"]["adam_epsilon"],
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["model"]["warmup_steps"],
            num_training_steps=config["model"]["max_step"],
        )
    else:
        for n, p in model.named_parameters():
            if "bert_policy" in n:
                p.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["model"]["learning_rate"],
        )

    max_step = config["model"]["max_step"]
    check_step = config["model"]["check_step"]
    batch_size = config["model"]["batch_size"]
    model.zero_grad()
    train_intent_loss = 0
    best_val_f1 = 0.0

    writer.add_text("config", json.dumps(config))

    for step in range(1, max_step + 1):
        model.train()
        batched_data = dataloader.get_train_batch(batch_size)
        batched_data = tuple(t.to(device) for t in batched_data)
        word_seq_tensor, word_mask_tensor, intent_tensor = batched_data
        intent_logits, intent_loss = model.forward(
            word_seq_tensor, word_mask_tensor, intent_tensor
        )

        train_intent_loss += intent_loss.item()
        loss = intent_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if config["model"]["finetune"]:
            scheduler.step()  # Update learning rate schedule

        model.zero_grad()
        if step % check_step == 0:
            train_intent_loss = train_intent_loss / check_step
            print("[%d|%d] step" % (step, max_step))
            print("\t intent loss:", train_intent_loss)

            predict_golden = {"intent": []}

            val_intent_loss = 0
            model.eval()
            for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(
                batch_size, data_key="val"
            ):
                pad_batch = tuple(t.to(device) for t in pad_batch)
                word_seq_tensor, word_mask_tensor, intent_tensor = pad_batch

                with torch.no_grad():
                    intent_logits, intent_loss = model.forward(
                        word_seq_tensor, word_mask_tensor, intent_tensor
                    )

                val_intent_loss += intent_loss.item() * real_batch_size
                for j in range(real_batch_size):
                    predicts = recover_intent(dataloader, intent_logits[j])
                    labels = ori_batch[j][1]

                    predict_golden["intent"].append(
                        {
                            "predict": [x for x in predicts],
                            "golden": [x for x in labels],
                        }
                    )

            total = len(dataloader.data["val"])
            val_intent_loss /= total
            print("%d samples val" % total)
            print("\t intent loss:", val_intent_loss)

            writer.add_scalar("intent_loss/train", train_intent_loss, global_step=step)
            writer.add_scalar("intent_loss/val", val_intent_loss, global_step=step)

            for x in ["intent"]:
                precision, recall, F1 = calculate_f1(predict_golden[x])
                print("-" * 20 + x + "-" * 20)
                print("\t Precision: %.2f" % (100 * precision))
                print("\t Recall: %.2f" % (100 * recall))
                print("\t F1: %.2f" % (100 * F1))

                writer.add_scalar(
                    "val_{}/precision".format(x), precision, global_step=step
                )
                writer.add_scalar("val_{}/recall".format(x), recall, global_step=step)
                writer.add_scalar("val_{}/F1".format(x), F1, global_step=step)

            if F1 > best_val_f1:
                best_val_f1 = F1
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "pytorch-intent-with-bert_policy.pt"),
                )
                print("best val F1 %.4f" % best_val_f1)
                print("save on", output_dir)

            train_intent_loss = 0

    writer.add_text("val intent F1", "%.2f" % (100 * best_val_f1))
    writer.close()

    model_path = os.path.join(output_dir, "pytorch-intent-with-bert_policy.pt")  ##存放模型
    zip_path = config["zipped_model_path"]
    zip_path = os.path.join(root_path, zip_path)
    print("zip model to", zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:  ##存放压缩模型
        zf.write(model_path)
