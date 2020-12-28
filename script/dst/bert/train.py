import os
import json
import time
import random
import warnings
from copy import deepcopy
from functools import partial
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.multiprocessing import Manager, Pool

from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification

from script.dst.bert.utils import eval_metrics, get_recall
from xbot.util.download import download_from_url
from xbot.util.path import get_data_path, get_root_path, get_config_path
from data.crosswoz.data_process.dst.bert_preprocess import (
    turn2example,
    DSTDataset,
    collate_fn,
)

warnings.simplefilter("ignore")


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["vocab"])
        self.ontology = json.load(
            open(config["cleaned_ontology"], "r", encoding="utf8")
        )
        start_time = time.time()
        self.train_dataloader = self.load_data(
            data_path=self.config["train4bert_dst"], data_type="train"
        )
        self.eval_dataloader = self.load_data(
            data_path=self.config["dev4bert_dst"], data_type="dev"
        )
        self.test_dataloader = self.load_data(
            data_path=self.config["test4bert_dst"], data_type="tests"
        )
        # 增加不使用下采样模拟推理过程，比较评价指标
        self.config["random_undersampling"] = 0
        self.config["overall_undersampling_ratio"] = 0.02  # 为了减少评估时间，可以根据自己的机器能力设定
        self.no_undersampling_test_dataloader = self.load_data(
            data_path=self.config["test4bert_dst"], data_type="tests"
        )
        elapsed = time.time() - start_time
        print(f"Loading data cost {elapsed}s ...")

        self.best_model_path = None
        self.model = None
        self.model_config = None
        self.optimizer = None

    def set_model(self) -> None:
        """init model, optimizer and training settings"""
        self.model_config = BertConfig.from_pretrained(self.config["config"])
        self.model = BertForSequenceClassification.from_pretrained(
            self.config["pytorch_model"], config=self.model_config
        )
        self.optimizer = opt.AdamW(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        self.model.to(self.config["device"])
        if self.config["n_gpus"] > 1:
            self.model = nn.DataParallel(self.model)

    @staticmethod
    def get_pos_neg_examples(
        example: tuple, pos_examples: List[tuple], neg_examples: List[tuple]
    ) -> None:
        """According to label saving example.

        Args:
            example: a new generated example, (input_ids, token_type_ids, domain, slot, value ...)
            pos_examples: all positive examples are saved in pos_examples
            neg_examples: all negative examples are saved in pos_examples
        """
        if example[-3] == 1:
            pos_examples.append(example)
        else:
            neg_examples.append(example)

    def iter_dials(
        self,
        dials: List[Tuple[str, dict]],
        data_type: str,
        pos_examples: List[tuple],
        neg_examples: List[tuple],
        process_id: int,
    ) -> None:
        """Iterate on dialogues, turns in one dialogue to generate examples

        Args:
            dials: raw dialogues data
            data_type: train, dev or tests
            pos_examples: all positive examples are saved in pos_examples
            neg_examples: all negative examples are saved in pos_examples
            process_id: current Process id
        """
        for dial_id, dial in tqdm(
            dials, desc=f"Building {data_type} examples, current process-{process_id}"
        ):
            sys_utter = "对话开始"
            for turn_id, turn in enumerate(dial["messages"]):
                if turn["role"] == "sys":
                    sys_utter = turn["content"]
                else:
                    raw_belief_state = turn["belief_state"]
                    belief_state = self.format_belief_state(raw_belief_state)

                    usr_utter = turn["content"]
                    context = sys_utter + self.tokenizer.sep_token + usr_utter
                    context_ids = self.tokenizer.encode(
                        context, add_special_tokens=False
                    )

                    cur_dialog_act = turn["dialog_act"]
                    triple_labels = self.format_labels(cur_dialog_act)

                    cur_pos_examples, cur_neg_examples = self.ontology2examples(
                        belief_state, context_ids, dial_id, triple_labels, turn_id
                    )

                    pos_examples.extend(cur_pos_examples)
                    neg_examples.extend(cur_neg_examples)

    @staticmethod
    def format_belief_state(
        raw_belief_state: List[Dict[str, list]]
    ) -> List[Tuple[str, str, str]]:
        """Reformat raw belief state to the format that model need.

        Args:
            raw_belief_state: e.g.
                                "belief_state": [
                                                  {
                                                    "slots": [
                                                      [
                                                        "餐馆-推荐菜",
                                                        "驴 杂汤"
                                                      ]
                                                    ]
                                                  }
                                                ]

        Returns:
            belief_state, reformatted belief state
        """
        belief_state = []
        for bs in raw_belief_state:
            domain, slot = bs["slots"][0][0].split("-")
            value = "".join(bs["slots"][0][1].split(" "))
            belief_state.append((domain, slot, value))
        return belief_state

    def ontology2examples(
        self,
        belief_state: List[Tuple[str, str, str]],
        context_ids: List[int],
        dial_id: str,
        triple_labels: Set[tuple],
        turn_id: int,
    ) -> Tuple[list, list]:
        """Iterate item in ontology to build examples.

        Args:
            belief_state: return value of method `format_belief_state`
            context_ids: context token's id in bert vocab
            dial_id: dialogue id in raw dialogue data
            triple_labels: triple label (domain, slot, value)
            turn_id: turn id in one dialogue

        Returns:
            new generated examples based on ontology
        """
        pos_examples = []
        neg_examples = []

        for (
            domain_slots,
            values,
        ) in self.ontology.items():
            domain_slot = domain_slots.split("-")
            domain, slot = domain_slot

            if domain in ["reqmore"]:
                continue

            if domain not in ["greet", "welcome", "thank", "bye"] and slot != "酒店设施":
                example = turn2example(
                    self.tokenizer,
                    domain,
                    "Request",
                    slot,
                    context_ids,
                    triple_labels,
                    belief_state,
                    dial_id,
                    turn_id,
                )
                self.get_pos_neg_examples(example, pos_examples, neg_examples)

            for value in values:
                value = "".join(value.split(" "))
                if slot == "酒店设施":
                    slot_value = slot + f"-{value}"
                    example = turn2example(
                        self.tokenizer,
                        domain,
                        "Request",
                        slot_value,
                        context_ids,
                        triple_labels,
                        belief_state,
                        dial_id,
                        turn_id,
                    )
                    self.get_pos_neg_examples(example, pos_examples, neg_examples)

                example = turn2example(
                    self.tokenizer,
                    domain,
                    slot,
                    value,
                    context_ids,
                    triple_labels,
                    belief_state,
                    dial_id,
                    turn_id,
                )

                self.get_pos_neg_examples(example, pos_examples, neg_examples)

        if self.config["random_undersampling"]:
            neg_examples = random.sample(
                neg_examples,
                k=self.config["neg_pos_sampling_ratio"] * len(pos_examples),
            )

        return pos_examples, neg_examples

    @staticmethod
    def format_labels(dialog_act: List[List[str]]) -> Set[tuple]:
        """Reformat raw dialog act to triple labels.

        Args:
            dialog_act: [
                          [
                            "Inform",
                            "餐馆",
                            "周边景点",
                            "小汤山现代农业科技示范园"
                          ],...
                        ]

        Returns:
            triple_labels, reformatted labels, (domain, slot, value)
        """
        turn_labels = dialog_act
        triple_labels = set()
        for usr_da in turn_labels:
            intent, domain, slot, value = usr_da
            if intent == "Request":
                triple_labels.add((domain, "Request", slot))
            else:
                if "-" in slot:  # 酒店设施
                    slot, value = slot.split("-")
                triple_labels.add((domain, slot, value))
        return triple_labels

    def build_examples(
        self, data_path: str, data_cache_path: str, data_type: str
    ) -> List[tuple]:
        """Generate data_type dataset and cache them.

        Args:
            data_path: raw dialogue data path
            data_cache_path: data save path
            data_type: train, dev or tests

        Returns:
            examples, mix up positive and negative examples
        """
        dials = json.load(open(data_path, "r", encoding="utf8"))
        dials = list(dials.items())
        if self.config["debug"]:
            dials = dials[: self.config["num_processes"]]

        num_orig_dials = len(dials)
        num_sampling_dials = int(
            len(dials) * self.config["overall_undersampling_ratio"]
        )
        dials = random.sample(dials, k=num_sampling_dials)
        print(
            f"After overall undersampling, {num_orig_dials} dialogues reduce to {num_sampling_dials} ..."
        )

        neg_examples, pos_examples = self.async_build_examples(data_type, dials)

        examples = pos_examples + neg_examples
        print(f"{len(dials)} dialogs generate {len(examples)} examples ...")

        random.shuffle(examples)
        examples = list(zip(*examples))
        torch.save(examples, data_cache_path)

        return examples

    def async_build_examples(
        self, data_type: str, dials: List[Tuple[str, dict]]
    ) -> Tuple[list, list]:
        """Use multiprocessing to process raw dialogue data.

        Args:
            data_type: train, dev or tests
            dials: raw dialogues data

        Returns:
            new examples by all processes
        """
        neg_examples = Manager().list()
        pos_examples = Manager().list()
        dials4single_process = (len(dials) - 1) // self.config["num_processes"] + 1
        print(f"Single process have {dials4single_process} dials ...")
        pool = Pool(self.config["num_processes"])
        for i in range(self.config["num_processes"]):
            pool.apply_async(
                func=self.iter_dials,
                args=(
                    dials[dials4single_process * i : dials4single_process * (i + 1)],
                    data_type,
                    pos_examples,
                    neg_examples,
                    i,
                ),
            )
        pool.close()
        pool.join()

        pos_examples = list(pos_examples)
        neg_examples = list(neg_examples)
        return neg_examples, pos_examples

    def load_data(self, data_path: str, data_type: str) -> DataLoader:
        """Loading data by loading cache data or generating examples from scratch.

        Args:
            data_path: raw dialogue data
            data_type: train, dev or tests

        Returns:
            dataloader, see torch.utils.data.DataLoader,
            https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20dataloader#torch.utils.data.DataLoader
        """
        print(f"Starting preprocess {data_type} data ...")
        raw_data_name = os.path.basename(data_path)
        processed_data_name = "processed_" + raw_data_name.split(".")[0] + ".pt"
        data_cache_path = os.path.join(self.config["data_path"], processed_data_name)
        if self.config["use_cache_data"] and os.path.exists(data_cache_path):
            print(f"Loading cache {data_type} data ...")
            examples = torch.load(data_cache_path)
            print(f"Total {len(examples[0])} {data_type} examples ...")
        else:
            examples = self.build_examples(data_path, data_cache_path, data_type)

        dataset = DSTDataset(examples)
        shuffle = True if data_type == "train" else False
        batch_size = self.config[f"{data_type}_batch_size"]
        collate = partial(collate_fn, mode=data_type)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config["num_workers"],
            collate_fn=collate,
        )
        return dataloader

    def evaluation(
        self, dataloader: DataLoader, epoch: Optional[int] = None, mode: str = "dev"
    ) -> float:
        """Evaluation on dev dataset or tests dataset.

        calculate `turn_inform` accuracy, `turn_request` accuracy and `joint_goal` accuracy

        Args:
            dataloader: see torch.utils.data.DataLoader
            epoch: current training epochs
            mode: train, dev or tests

        Returns:
            specified metric value
        """
        self.model.eval()
        eval_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Evaluating" if mode == "dev" else "Testing",
        )
        results = defaultdict(dict)
        with torch.no_grad():
            for step, batch in eval_bar:
                inputs = {
                    k: v.to(self.config["device"]) for k, v in list(batch.items())[:4]
                }
                loss, logits = self.model(**inputs)[:2]

                if self.config["n_gpus"] > 1:
                    loss = loss.mean()

                # preds = logits.argmax(dim=-1).cpu().tolist()
                max_logits, preds = [item.cpu().tolist() for item in logits.max(dim=1)]

                labels = inputs["labels"].cpu().tolist()

                self.format_results(batch, labels, preds, max_logits, results)

                desc = (
                    f"Evaluating： Epoch: {epoch}, " if mode == "dev" else "Best model, "
                )
                desc += f"CELoss: {loss.item():.3f}"
                eval_bar.set_description(desc)

        metrics_res = eval_metrics(
            results, self.config["data_path"], self.config["top_k"]
        )
        print("*" * 10 + " eval metrics " + "*" * 10)
        print(json.dumps(metrics_res, indent=2))
        return metrics_res[self.config["eval_metric"]]

    @staticmethod
    def format_results(
        batch: Dict[str, torch.Tensor],
        labels: List[int],
        preds: List[int],
        logits: List[float],
        results: Dict[str, dict],
    ) -> None:
        """Reformat evaluation results to facilitate the calculation of metrics.

        Args:
            batch: batch data
            labels: ground truth
            preds: model output
            logits: preds corresponding logits
            results: save labels and pred based on dialogue id, turn id
        """
        for i, (logit, pred, label, belief_state, dialogue_idx, turn_id) in enumerate(
            zip(
                logits,
                preds,
                labels,
                batch["belief_states"],
                batch["dialogue_idxs"],
                batch["turn_ids"],
            )
        ):
            if turn_id not in results[dialogue_idx]:
                results[dialogue_idx][turn_id] = {}
            if "preds" not in results[dialogue_idx][turn_id]:
                results[dialogue_idx][turn_id] = {
                    "logits": [],
                    "preds": [],
                    "labels": [],
                    "belief_state": belief_state,
                }

            triple = (batch["domains"][i], batch["slots"][i], batch["values"][i])
            if pred == 1:
                results[dialogue_idx][turn_id]["preds"].append(triple)
                results[dialogue_idx][turn_id]["logits"].append(logit)
            if label == 1:
                results[dialogue_idx][turn_id]["labels"].append(triple)

    def eval_test(self) -> None:
        """Loading best model to evaluate tests dataset."""
        if self.best_model_path is not None:
            if hasattr(self.model, "module"):
                self.model.module = BertForSequenceClassification.from_pretrained(
                    self.best_model_path
                )
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    self.best_model_path
                )
            self.model.to(self.config["device"])
        self.evaluation(self.test_dataloader, mode="tests")
        self.evaluation(self.no_undersampling_test_dataloader, mode="tests")

    def train(self) -> None:
        """Training."""
        self.set_model()
        epoch_bar = trange(0, self.config["num_epochs"], desc="Epoch")
        best_metric = 0

        for epoch in epoch_bar:
            self.model.train()
            train_bar = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc="Training",
            )
            for step, batch in train_bar:
                inputs = {k: v.to(self.config["device"]) for k, v in batch.items()}
                loss = self.model(**inputs)[0]

                if self.config["n_gpus"] > 1:
                    loss = loss.mean()

                self.model.zero_grad()
                loss.backward()
                clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config["max_grad_norm"]
                )
                self.optimizer.step()

                train_bar.set_description(
                    f"Training： Epoch: {epoch}, Iter: {step}, CELoss: {loss.item():.3f}"
                )

            eval_metric = self.evaluation(self.eval_dataloader, epoch)
            if eval_metric > best_metric:
                print(
                    f'Best model saved, {self.config["eval_metric"]}: {eval_metric} ...'
                )
                best_metric = eval_metric
                self.save(epoch, best_metric)

    def save(self, epoch: int, best_metric: float) -> None:
        """Save the best model according to specified metric.

        Args:
            epoch: current training epoch
            best_metric: specified metric
        """
        save_name = f'Epoch-{epoch}-{self.config["eval_metric"]}-{best_metric:.3f}'
        self.best_model_path = os.path.join(self.config["output_dir"], save_name)
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)

        model_to_save = deepcopy(
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.cpu().save_pretrained(self.best_model_path)
        self.tokenizer.save_pretrained(self.best_model_path)


def main():
    model_config_name = "dst/bert/train.json"
    common_config_name = "dst/bert/common.json"

    data_urls = {
        "train4bert_dst.json": "http://xbot.bslience.cn/train4bert_dst.json",
        "dev4bert_dst.json": "http://xbot.bslience.cn/dev4bert_dst.json",
        "test4bert_dst.json": "http://xbot.bslience.cn/test4bert_dst.json",
        "cleaned_ontology.json": "http://xbot.bslience.cn/cleaned_ontology.json",
        "config.json": "http://xbot.bslience.cn/bert-base-chinese/config.json",
        "pytorch_model.bin": "http://xbot.bslience.cn/bert-base-chinese/pytorch_model.bin",
        "vocab.txt": "http://xbot.bslience.cn/bert-base-chinese/vocab.txt",
    }

    # load config
    root_path = get_root_path()
    common_config_path = os.path.join(get_config_path(), common_config_name)
    train_config_path = os.path.join(get_config_path(), model_config_name)
    common_config = json.load(open(common_config_path))
    train_config = json.load(open(train_config_path))
    train_config.update(common_config)
    train_config["n_gpus"] = torch.cuda.device_count()
    train_config["train_batch_size"] = (
        max(1, train_config["n_gpus"]) * train_config["train_batch_size"]
    )
    train_config["device"] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_config["data_path"] = os.path.join(get_data_path(), "crosswoz/dst_bert_data")
    train_config["output_dir"] = os.path.join(root_path, train_config["output_dir"])
    if not os.path.exists(train_config["data_path"]):
        os.makedirs(train_config["data_path"])
    if not os.path.exists(train_config["output_dir"]):
        os.makedirs(train_config["output_dir"])

    # download data
    for data_key, url in data_urls.items():
        dst = os.path.join(train_config["data_path"], data_key)
        file_name = data_key.split(".")[0]
        train_config[file_name] = dst
        if not os.path.exists(dst):
            download_from_url(url, dst)

    # train
    trainer = Trainer(train_config)
    trainer.train()
    trainer.eval_test()
    get_recall(train_config["data_path"])


if __name__ == "__main__":
    main()
