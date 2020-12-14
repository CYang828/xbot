import os
import json
from functools import partial
from typing import List

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, BertTokenizer

from xbot.util.dst_util import DST
from xbot.util.state import default_state
from xbot.util.download import download_from_url
from xbot.util.path import get_data_path, get_root_path, get_config_path
from data.crosswoz.data_process.dst.bert_preprocess import (
    turn2example,
    DSTDataset,
    collate_fn,
    rank_values,
)


class BertDST(DST):
    infer_config_name = "dst/bert/inference.json"
    common_config_name = "dst/bert/common.json"

    data_urls = {
        "cleaned_ontology.json": "http://xbot.bslience.cn/cleaned_ontology.json",
        "config.json": "http://xbot.bslience.cn/bert-dst/config.json",
        "pytorch_model.bin": "http://xbot.bslience.cn/bert-dst/pytorch_model.bin",
        "vocab.txt": "http://xbot.bslience.cn/bert-dst/vocab.txt",
    }

    def __init__(self):
        super(BertDST, self).__init__()
        # load config
        infer_config = self.load_config()

        # download data
        self.download_data(infer_config)

        self.ontology = json.load(
            open(infer_config["cleaned_ontology"], "r", encoding="utf8")
        )
        self.model = BertForSequenceClassification.from_pretrained(
            infer_config["model_dir"]
        )
        self.model.to(infer_config["device"])
        self.tokenizer = BertTokenizer.from_pretrained(infer_config["model_dir"])
        self.config = infer_config

        self.model.eval()
        self.state = default_state()
        self.domains = set(self.state["belief_state"].keys())

    @staticmethod
    def download_data(infer_config: dict) -> None:
        """Download trained model and ontology file for inference.

        Args:
            infer_config: config used for inference
        """
        for data_key, url in BertDST.data_urls.items():
            if "ontology" in data_key:
                dst = os.path.join(infer_config["data_path"], data_key)
            else:
                model_dir = os.path.join(infer_config["data_path"], "trained_model")
                infer_config["model_dir"] = model_dir
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                dst = os.path.join(model_dir, data_key)
            file_name = data_key.split(".")[0]
            infer_config[file_name] = dst
            if not os.path.exists(dst):
                download_from_url(url, dst)

    @staticmethod
    def load_config() -> dict:
        """Load config from common config and inference config from xbot/config/dst/bert .

        Returns:
            config dict
        """
        root_path = get_root_path()
        common_config_path = os.path.join(get_config_path(), BertDST.common_config_name)
        infer_config_path = os.path.join(get_config_path(), BertDST.infer_config_name)
        common_config = json.load(open(common_config_path))
        infer_config = json.load(open(infer_config_path))
        infer_config.update(common_config)
        infer_config["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        infer_config["data_path"] = os.path.join(
            get_data_path(), "crosswoz/dst_bert_data"
        )
        infer_config["output_dir"] = os.path.join(root_path, infer_config["output_dir"])
        if not os.path.exists(infer_config["data_path"]):
            os.makedirs(infer_config["data_path"])
        if not os.path.exists(infer_config["output_dir"]):
            os.makedirs(infer_config["output_dir"])
        return infer_config

    def preprocess(self, sys_uttr: str, usr_uttr: str) -> DataLoader:
        """Preprocess raw utterance, convert them to token id for forward.

        Args:
            sys_uttr: response of previous system turn
            usr_uttr: previous turn user's utterance

        Returns:
            DataLoader for inference
        """
        context = sys_uttr + self.tokenizer.sep_token + usr_uttr
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)

        examples = self.build_examples(context_ids)

        # # 如果知道当前 domain，可以仅 request 当前 domain，这里只能随机采样
        # request_examples = []
        # no_request_examples = []
        # for example in examples:
        #     if example[-2] == 'Request':
        #         request_examples.append(example)
        #     else:
        #         no_request_examples.append(example)
        # request_examples = random.sample(request_examples, k=int(0.2 * len(request_examples)))
        # examples = request_examples + no_request_examples
        random.shuffle(examples)

        examples = list(zip(*examples))
        dataset = DSTDataset(examples)
        collate = partial(collate_fn, mode="infer")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config["batch_size"],
            collate_fn=collate,
            shuffle=False,
            num_workers=self.config["num_workers"],
        )
        return dataloader

    def build_examples(self, context_ids: List[int]) -> List[tuple]:
        """Build examples according to ontology.

        Args:
            context_ids: dialogue history id based on BertTokenizer

        Returns:
            a list of example, (input_ids, token_type_ids, domain, slot, value)
        """
        examples = []
        for domain_slots, values in self.ontology.items():
            domain, slot = domain_slots.split("-")

            if domain in ["reqmore"]:
                continue

            if domain not in ["greet", "welcome", "thank", "bye"] and slot != "酒店设施":
                example = turn2example(
                    self.tokenizer, domain, "Request", slot, context_ids
                )
                examples.append(example)

            for value in values:
                value = "".join(value.split(" "))
                if slot == "酒店设施":
                    slot_value = slot + f"-{value}"
                    example = turn2example(
                        self.tokenizer, domain, "Request", slot_value, context_ids
                    )
                    examples.append(example)

                example = turn2example(self.tokenizer, domain, slot, value, context_ids)
                examples.append(example)
        return examples

    def init_session(self) -> None:
        """Initiate state of one session. """
        self.state = default_state()

    def update(self, action: List[tuple]) -> None:
        """Update session's state according to output of bert.

        Args:
            action: output of NLU module, but in bert dst, inputs are utterance of user and system,
                    action is not used
        """
        usr_utter = self.state["history"][-1][1]
        usr_utter = "".join(usr_utter.split())
        sys_uttr = ""
        if len(self.state["history"]) > 1:
            sys_uttr = self.state["history"][-2][1]
            sys_uttr = "".join(sys_uttr.split())

        # forward
        pred_labels = self.forward(sys_uttr, usr_utter)

        # update
        self.update_state(pred_labels)

    def update_state(self, pred_labels: List[tuple]) -> None:
        """Update request slots and belief state in state.

        Args:
            pred_labels: triple labels, (domain, slot, value)
        """
        for domain, slot, value in pred_labels:
            if slot == "Request":
                self.state["request_slots"].append([domain, value])
            else:
                if domain not in self.domains:
                    continue
                if slot in self.state["belief_state"][domain]:
                    self.state["belief_state"][domain][slot] = value

    def forward(self, sys_uttr: str, usr_utter: str) -> List[tuple]:
        """Bert model forward and rank output triple labels.

        Args:
            sys_uttr: response of previous system turn
            usr_utter: previous turn user's utterance

        Returns:
            a list of triple labels, (domain, slot, value)
        """
        pred_labels = []
        true_logits = []
        dataloader = self.preprocess(sys_uttr, usr_utter)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Inferring")

        with torch.no_grad():
            for step, batch in pbar:
                inputs = {
                    k: v.to(self.config["device"]) for k, v in list(batch.items())[:3]
                }
                logits = self.model(**inputs)[0]

                max_logits, preds = [item.cpu().tolist() for item in logits.max(dim=1)]

                for i, (pred, logit) in enumerate(zip(preds, max_logits)):
                    triple = (
                        batch["domains"][i],
                        batch["slots"][i],
                        batch["values"][i],
                    )
                    if pred == 1:
                        true_logits.append(logit)
                        pred_labels.append(triple)

        pred_labels = rank_values(true_logits, pred_labels, self.config["top_k"])

        return pred_labels


if __name__ == "__main__":
    import random

    dst_model = BertDST()
    data_path = os.path.join(get_data_path(), "crosswoz/dst_trade_data")
    with open(os.path.join(data_path, "test_dials.json"), "r", encoding="utf8") as f:
        dials = json.load(f)
        example = random.choice(dials)
        break_turn = 0
        for ti, turn in enumerate(example["dialogue"]):
            dst_model.state["history"].append(("sys", turn["system_transcript"]))
            dst_model.state["history"].append(("usr", turn["transcript"]))
            dst_model.update([])
            if random.random() < 0.5:
                break_turn = ti + 1
                break
    if break_turn == len(example["dialogue"]):
        print("对话已完成，请重新开始测试")
    print("对话状态更新后：")
    print(json.dumps(dst_model.state, indent=2, ensure_ascii=False))
