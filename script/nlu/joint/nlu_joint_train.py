import os
import json
import random
import numpy as np

from xbot.util.path import get_root_path
from xbot.util.download import download_from_url
from xbot.nlu.joint.joint_with_bert import JointWithBert
from data.crosswoz.data_process.nlu_dataloader import Dataloader
from data.crosswoz.data_process.nlu_postprocess import (
    is_slot_da,
    calculateF1,
    recover_intent,
)

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    data_urls = {
        "joint_train_data.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/joint_train_data.json",
        "joint_val_data.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/slot_val_data.json",
        "joint_test_data.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/joint_test_data.json",
    }

    # load config
    root_path = get_root_path()
    config_path = os.path.join(
        root_path, "xbot/config/crosswoz_all_context_nlu_slot.json"
    )
    config = json.load(open(config_path))
    data_path = config["data_dir"]
    data_path = os.path.join(root_path, data_path)
    output_dir = config["output_dir"]
    output_dir = os.path.join(root_path, output_dir)
    log_dir = config["log_dir"]
    output_dir = os.path.join(root_path, output_dir)
    device = config["DEVICE"]

    # download data
    for data_key, url in data_urls.items():
        dst = os.path.join(os.path.join(data_path, data_key))
        if not os.path.exists(dst):
            download_from_url(url, dst)

    set_seed(config["seed"])

    # 经过preprocess处理之后，会产生intent_vocab，tag_vocab，train_data,test_data,val_data,数据集
    # 导入intent_vocab，tag_vocab数据集
    intent_vocab = json.load(open(os.path.join(root_path, "intent_vocab.json")))
    tag_vocab = json.load(open(os.path.join(root_path, "tag_vocab.json")))

    dataloader = Dataloader(
        intent_vocab=intent_vocab,
        tag_vocab=tag_vocab,
        pretrained_weights=config["model"]["pretrained_weights"],
    )

    # 导入train_data,val_data,test_data数据集
    for data_key in ["train", "val", "tests"]:
        dataloader.load_data(
            json.load(open(os.path.join(root_path, "{}_data.json".format(data_key)))),
            data_key,
            cut_sen_len=config["cut_sen_len"],
            use_bert_tokenizer=config["use_bert_tokenizer"],
        )
        print("{} set size: {}".format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)
    # 导入模型
    model = JointWithBert(
        config["model"],
        device,
        dataloader.tag_dim,
        dataloader.intent_dim,
        dataloader.intent_weight,
    )
    model.to(device)

    # 判断是否进行finetune
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

    for name, param in model.named_parameters():
        print(name, param.shape, param.device, param.requires_grad)

    max_step = config["model"]["max_step"]
    check_step = config["model"]["check_step"]
    batch_size = config["model"]["batch_size"]
    model.zero_grad()
    train_slot_loss, train_intent_loss = 0, 0
    best_val_f1 = 0.0

    writer.add_text("config", json.dumps(config))

    for step in range(1, max_step + 1):
        model.train()
        batched_data = dataloader.get_train_batch(batch_size)  # 随机获得batch_size个样本,
        # 因为在调用get_train_batch的时候，需要调用pad_batch，所以得到的输出是7维的
        batched_data = tuple(t.to(device) for t in batched_data)
        (
            word_seq_tensor,
            tag_seq_tensor,
            intent_tensor,
            word_mask_tensor,
            tag_mask_tensor,
            context_seq_tensor,
            context_mask_tensor,
        ) = batched_data
        if not config["model"]["context"]:
            context_seq_tensor, context_mask_tensor = None, None
        _, _, slot_loss, intent_loss = model(
            word_seq_tensor,
            word_mask_tensor,
            tag_seq_tensor,
            tag_mask_tensor,
            intent_tensor,
            context_seq_tensor,
            context_mask_tensor,
        )
        train_slot_loss += slot_loss.item()
        train_intent_loss += intent_loss.item()
        loss = slot_loss + intent_loss  # 将slot_loss和intent_loss直接相加
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 进行梯度裁剪，防止梯度爆炸
        optimizer.step()
        if config["model"]["finetune"]:
            # Update learning rate schedule
            scheduler.step()
        model.zero_grad()  #
        if step % check_step == 0:
            train_slot_loss = train_slot_loss / check_step
            train_intent_loss = train_intent_loss / check_step
            print("[%d|%d] step" % (step, max_step))
            print("\t slot loss:", train_slot_loss)
            print("\t intent loss:", train_intent_loss)

            predict_golden = {"intent": [], "slot": [], "overall": []}

            val_slot_loss, val_intent_loss = 0, 0
            model.eval()
            for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(
                batch_size, data_key="val"
            ):
                pad_batch = tuple(t.to(device) for t in pad_batch)
                (
                    word_seq_tensor,
                    tag_seq_tensor,
                    intent_tensor,
                    word_mask_tensor,
                    tag_mask_tensor,
                    context_seq_tensor,
                    context_mask_tensor,
                ) = pad_batch
                if not config["model"]["context"]:
                    context_seq_tensor, context_mask_tensor = None, None

                with torch.no_grad():
                    slot_logits, intent_logits, slot_loss, intent_loss = model.forward(
                        word_seq_tensor,
                        word_mask_tensor,
                        tag_seq_tensor,
                        tag_mask_tensor,
                        intent_tensor,
                        context_seq_tensor,
                        context_mask_tensor,
                    )
                val_slot_loss += slot_loss.item() * real_batch_size
                val_intent_loss += intent_loss.item() * real_batch_size
                for j in range(real_batch_size):
                    predicts = recover_intent(
                        dataloader,
                        intent_logits[j],
                        slot_logits[j],
                        tag_mask_tensor[j],
                        ori_batch[j][0],
                        ori_batch[j][-4],
                    )
                    labels = ori_batch[j][3]

                    predict_golden["overall"].append(
                        {"predict": predicts, "golden": labels}
                    )
                    predict_golden["slot"].append(
                        {
                            "predict": [x for x in predicts if is_slot_da(x)],
                            "golden": [x for x in labels if is_slot_da(x)],
                        }
                    )
                    predict_golden["intent"].append(
                        {
                            "predict": [x for x in predicts if not is_slot_da(x)],
                            "golden": [x for x in labels if not is_slot_da(x)],
                        }
                    )

            for j in range(10):
                writer.add_text(
                    "val_sample_{}".format(j),
                    json.dumps(
                        predict_golden["overall"][j], indent=2, ensure_ascii=False
                    ),
                    global_step=step,
                )

            total = len(dataloader.data["val"])
            val_slot_loss /= total
            val_intent_loss /= total
            print("%d samples val" % total)
            print("\t slot loss:", val_slot_loss)
            print("\t intent loss:", val_intent_loss)

            writer.add_scalar("intent_loss/train", train_intent_loss, global_step=step)
            writer.add_scalar("intent_loss/val", val_intent_loss, global_step=step)

            writer.add_scalar("slot_loss/train", train_slot_loss, global_step=step)
            writer.add_scalar("slot_loss/val", val_slot_loss, global_step=step)

            for x in ["intent", "slot", "overall"]:
                precision, recall, F1 = calculateF1(predict_golden[x])
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
                    model.state_dict(), os.path.join(output_dir, "pytorch_model_nlu.pt")
                )
                print("best val F1 %.4f" % best_val_f1)
                print("save on", output_dir)

            train_slot_loss, train_intent_loss = 0, 0

    writer.add_text("val overall F1", "%.2f" % (100 * best_val_f1))
    writer.close()
    model_path = os.path.join(output_dir, "pytorch_model_nlu.pt")
    torch.save(model.state_dict(), model_path)
