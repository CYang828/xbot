import os
import json
import copy
import random

from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from xbot.dm.dst.trade_dst.trade import Trade
from xbot.util.download import download_from_url
from xbot.util.path import get_root_path, get_config_path, get_data_path
from script.dst.trade.utils import (
    masked_cross_entropy_for_value,
    evaluate_metrics,
    prepare_data_seq,
)


class Trainer:
    def __init__(self, config, langs, gating_dict, slots):
        self.lang = langs[0]
        self.mem_lang = langs[1]
        self.lr = config["learning_rate"]
        self.n_gpus = config["n_gpus"]
        self.gating_dict = gating_dict
        self.num_gates = len(gating_dict)
        self.use_gate = config["use_gate"]
        self.output_dir = config["output_dir"]
        self.teacher_forcing_ratio = config["teacher_forcing_ratio"]

        self.cross_entropy = nn.CrossEntropyLoss()

        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = (
            0,
            1,
            0,
            0,
            0,
        )
        self.loss_grad, self.loss_ptr_to_bp = None, None

        self.model = Trade(
            lang=self.lang,
            vocab_size=len(self.lang.index2word),
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            pad_id=config["pad_id"],
            slots=slots,
            num_gates=len(gating_dict),
            unk_mask=config["unk_mask"],
            pretrained_embedding_path=config["pretrained_embedding_path"],
            load_embedding=config["load_embedding"],
            fix_embedding=config["fix_embedding"],
            parallel_decode=config["parallel_decode"],
        )

        model_path = config["model_path"]
        if model_path:
            print(f"Model {model_path} loaded")
            trained_model_params = torch.load(model_path)
            self.model.load_state_dict(trained_model_params)

        # Initialize optimizers and criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=0.0001,
            verbose=True,
        )

        self.reset()
        self.model.to(config["device"])
        if self.n_gpus > 1:
            self.model = nn.DataParallel(self.model)

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        self.print_every += 1
        return "L:{:.2f},LP:{:.2f},LG:{:.2f}".format(
            print_loss_avg, print_loss_ptr, print_loss_gate
        )

    def save_model(self, metric):
        name = f"trade-{metric}.pth"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save = copy.deepcopy(model_to_save)
        torch.save(
            model_to_save.cpu().state_dict(), os.path.join(self.output_dir, name)
        )

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = (
            0,
            1,
            0,
            0,
            0,
        )

    def train_batch(self, data, slots, reset=0):
        if reset:
            self.reset()
        self.optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        # (num_slots, bs, max_res_len, vocab_size), (num_slots, bs, num_gates), (num_slots, max_res_len, bs)
        all_point_outputs, gates = self.model(data, use_teacher_forcing, slots)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.contiguous(),
            data["generate_y"].contiguous(),  # [:,:len(self.point_slots)].contiguous(),
            data["y_lengths"],
        )  # [:,:len(self.point_slots)])
        loss_gate = self.cross_entropy(
            gates.contiguous().view(-1, gates.size(-1)),
            data["gating_label"].contiguous().view(-1),
        )

        if self.use_gate:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr

        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, grad_clip):
        self.loss_grad.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

    def evaluate(self, dev, metric_best, slot_temp, epoch, early_stop=None):
        # Set to not-training mode to disable dropout
        print("Starting Evaluation")

        self.model.eval()
        all_prediction = {}
        id2gate = {v: k for k, v in self.gating_dict.items()}
        progress_bar = tqdm(enumerate(dev), total=len(dev))

        for j, data_dev in progress_bar:
            all_point_outputs, gates = self.model(data_dev, False, slot_temp)
            batch_size, num_slots, max_res_len, vocab_size = all_point_outputs.size()
            words_id = all_point_outputs.argmax(dim=-1)
            words = []

            for bs in range(batch_size):
                bs_words = []
                for slot_id in range(num_slots):
                    one_words = [
                        self.lang.index2word[word_id.item()]
                        for word_id in words_id[bs][slot_id]
                    ]
                    bs_words.append(one_words)
                words.append(bs_words)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction:
                    all_prediction[data_dev["ID"][bi]] = {}

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                    "turn_belief": data_dev["turn_belief"][bi]
                }

                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                # (num_slots, bs, num_gates)
                gate = torch.argmax(gates[bi], dim=1)  # （num_slots,）

                # pointer-generator results
                if self.use_gate:
                    for si, sg in enumerate(gate):  # (num_slots, )
                        if sg == self.gating_dict["none"]:
                            continue
                        elif sg == self.gating_dict["ptr"]:
                            pred = words[bi][si]
                            st = []
                            for e in pred:
                                if e == "EOS":
                                    break
                                else:
                                    st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(
                                    slot_temp[si] + "-" + str(st)
                                )
                        else:  # dont care
                            predict_belief_bsz_ptr.append(
                                slot_temp[si] + "-" + id2gate[sg.item()]
                            )
                else:
                    for si, _ in enumerate(gate):
                        pred = words[bi][si]
                        st = []
                        for e in pred:
                            if e == "EOS":
                                break
                            else:
                                st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
                # 在 batch 上循环，分别以 dialogue_id, turn_id 为 key 添加值
                # all_prediction = {
                #       'dialogue_id': {
                #             'turn_id': {
                #                   'turn_belief': List['slot-value']
                #                   'pred_bs_ptr': List['slot-value']
                #             }
                #       }
                # }
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]][
                    "pred_bs_ptr"
                ] = predict_belief_bsz_ptr

        joint_acc_score_ptr, f1_score_ptr, turn_acc_score_ptr = evaluate_metrics(
            all_prediction, "pred_bs_ptr", slot_temp
        )

        evaluation_metrics = {
            "Joint Acc": joint_acc_score_ptr,
            "Turn Acc": turn_acc_score_ptr,
            "Joint F1": f1_score_ptr,
        }
        print("eval metrics:")
        print(evaluation_metrics)

        # Set back to training mode
        self.model.train()

        # (joint_acc_score_ptr + joint_acc_score_class) / 2
        joint_acc_score = joint_acc_score_ptr
        f1_score = f1_score_ptr

        if early_stop == "F1":
            if f1_score >= metric_best:
                self.save_model(f"Epoch-{epoch:d}-F1-{f1_score:.4f}")
                print("Model Saved")
            return f1_score
        else:
            if joint_acc_score >= metric_best:
                self.save_model(f"Epoch-{epoch:d}-JACC-{joint_acc_score:.4f}")
                print("Model Saved")
            return joint_acc_score


def main():
    model_config_name = "dst/trade/train.json"
    common_config_name = "dst/trade/common.json"

    data_urls = {
        "train_dials.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/train_dials.json",
        "dev_dials.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/dev_dials.json",
        "test_dials.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/test_dials.json",
        "ontology.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/ontology.json",
        "sgns.wiki.bigram.bz2": "http://qiw2jpwfc.hn-bkt.clouddn.com/sgns.wiki.bigram.bz2",
    }

    # load config
    root_path = get_root_path()
    common_config_path = os.path.join(get_config_path(), common_config_name)
    model_config_path = os.path.join(get_config_path(), model_config_name)
    common_config = json.load(open(common_config_path))
    model_config = json.load(open(model_config_path))
    model_config.update(common_config)
    model_config["n_gpus"] = torch.cuda.device_count()
    model_config["batch_size"] = (
        max(1, model_config["n_gpus"]) * model_config["batch_size"]
    )
    model_config["device"] = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    if model_config["load_embedding"]:
        model_config["hidden_size"] = 300

    model_config["data_path"] = os.path.join(get_data_path(), "crosswoz/dst_trade_data")
    model_config["output_dir"] = os.path.join(
        root_path, model_config["output_dir"]
    )  # 可以用来保存模型文件
    if model_config["load_model_name"]:
        model_config["model_path"] = os.path.join(
            model_config["output_dir"], model_config["load_model_name"]
        )
    else:
        model_config["model_path"] = ""
    if not os.path.exists(model_config["data_path"]):
        os.makedirs(model_config["data_path"])
    if not os.path.exists(model_config["output_dir"]):
        os.makedirs(model_config["output_dir"])

    # download data
    for data_key, url in data_urls.items():
        dst = os.path.join(model_config["data_path"], data_key)
        if "_" in data_key:
            file_name = data_key.split(".")[0]
        elif "wiki.bigram" in data_key:
            file_name = "orig_pretrained_embedding"
        else:
            file_name = data_key.split(".")[0]  # ontology
        model_config[file_name] = dst
        if not os.path.exists(dst):
            download_from_url(url, dst)

    avg_best, cnt, acc = 0.0, 0, 0.0

    # 数据预处理
    train, dev, test, langs, slots, gating_dict = prepare_data_seq(model_config)
    lang = langs[0]
    model_config["pretrained_embedding_path"] = os.path.join(
        model_config["data_path"], f"emb{len(lang.index2word)}"
    )

    print(f">>> Train configs:")
    print("\t", model_config)

    # 初始化训练
    trainer = Trainer(
        config=model_config, langs=langs, gating_dict=gating_dict, slots=slots
    )

    # 训练
    start_epoch = (
        0
        if not model_config["model_path"]
        else int(model_config["model_path"].split("-")[2]) + 1
    )

    for epoch in tqdm(range(start_epoch, model_config["num_epochs"]), desc="Epoch"):
        progress_bar = tqdm(enumerate(train), total=len(train))

        for i, data in progress_bar:
            trainer.train_batch(data, slots, reset=(i == 0))
            trainer.optimize(int(model_config["grad_clip"]))
            progress_bar.set_description(trainer.print_loss())

        if (epoch + 1) % int(model_config["eval_steps"]) == 0:

            acc = trainer.evaluate(
                dev, avg_best, slots, epoch, model_config["early_stop"]
            )
            trainer.scheduler.step(acc)

            if acc >= avg_best:
                avg_best = acc
                cnt = 0
            else:
                cnt += 1

            if cnt == model_config["patience"] or (
                acc == 1.0 and model_config["early_stop"] is None
            ):
                print("Ran out of patient, early stop...")
                break


if __name__ == "__main__":
    main()
