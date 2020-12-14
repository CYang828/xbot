import os
import time
import json
import random
from copy import deepcopy
from functools import partial
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as opt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from xbot.util.path import get_data_path
from xbot.util.train_util import update_config
from xbot.util.download import download_from_url
from xbot.util.file_util import load_json, dump_json
from data.crosswoz.data_process.policy.bert_proprecess import (
    PolicyDataset,
    collate_fn,
    str2id,
)
from script.policy.bert.utils import (
    preprocess,
    ACT_ONTOLOGY,
    eval_metrics,
    NUM_ACT,
    get_joint_acc,
)


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        model_config = BertConfig.from_pretrained(config["config"])
        model_config.num_labels = NUM_ACT
        self.model = BertForSequenceClassification.from_pretrained(
            config["pytorch_model"], config=model_config
        )
        self.tokenizer = BertTokenizer.from_pretrained(config["vocab"])

        start_time = time.time()
        self.train_dataloader = self.load_data("train")
        self.eval_dataloader = self.load_data("val")
        self.test_dataloader = self.load_data("tests")
        elapsed_time = time.time() - start_time
        print(f"Loading data cost {elapsed_time}s ...")

        self.optimizer = opt.AdamW(self.model.parameters(), lr=config["learning_rate"])
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.model.to(config["device"])
        if config["n_gpus"] > 1:
            self.model = nn.DataParallel(self.model)

        self.threshold = config["threshold"]
        self.best_model_path = None

    def load_data(self, data_type: str) -> DataLoader:
        """Load data from data cache or build from scratch.

        Args:
            data_type: train, dev or tests

        Returns:
            DataLoader, see torch.utils.data.DataLoader
        """
        raw_data_path = os.path.join(
            self.config["raw_data_path"], f"{data_type}.json.zip"
        )
        filename = f"{data_type}.json"
        output_path = os.path.join(self.config["data_path"], filename)
        if not os.path.exists(output_path) or not self.config["use_data_cache"]:
            examples = preprocess(raw_data_path, output_path, filename)
        else:
            print(f"Loading {data_type} data from cache ...")
            examples = load_json(output_path)

        if self.config["debug"]:
            examples = random.sample(examples, k=int(len(examples) * 0.1))
        examples_dict = self.get_input_ids(examples)

        print(f"Starting building {data_type} dataset ...")
        dataset = PolicyDataset(**examples_dict)
        shuffle = True if data_type == "train" else False
        collate = partial(collate_fn, mode=data_type)
        batch_size = self.config[f"{data_type}_batch_size"]
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.config["num_workers"],
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=collate,
        )
        return dataloader

    def get_input_ids(self, examples: List[dict]) -> Dict[str, List[list]]:
        """Convert utterance string to id based on BertTokenizer.

        Args:
            examples: a list of example, one example contain dialogue id, turn id, input id,
                      token type id, ground truth label

        Returns:
            examples dict, same type data are placed in a same list. e.g. dialogue id of all examples
            are placed into examples_dict['dial_ids']
        """
        examples_dict = defaultdict(list)

        for example in examples:
            input_ids, token_type_ids = str2id(
                self.tokenizer,
                example["sys_utter"],
                example["usr_utter"],
                example["source"],
            )

            examples_dict["dial_ids"].append(example["dial_id"])
            examples_dict["turn_ids"].append(example["turn_id"])
            examples_dict["input_ids"].append(input_ids)
            examples_dict["token_type_ids"].append(token_type_ids)
            examples_dict["labels"].append(example["label"])

        return examples_dict

    def eval_forward(
        self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluation forward step, calculate loss and probability of classes

        Args:
            batch: batch data, see output of collate_fn

        Returns:
            ground truth labels, forward loss, model prediction
        """
        inputs = {k: v.to(self.config["device"]) for k, v in list(batch.items())[:4]}
        labels = inputs.pop("labels")
        logits = self.model(**inputs)[0]
        loss = self.loss_fct(logits, labels)
        if self.config["n_gpus"] > 1:
            loss = loss.mean()
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        return labels, loss, preds

    @staticmethod
    def format_results(
        batch: dict, labels: torch.Tensor, preds: torch.Tensor, results: dict
    ) -> None:
        """Reformat evaluation results to facilitate the calculation of metrics.

        Args:
            batch: batch data
            labels: ground truth
            preds: model outputs
            results: preds corresponding logits
        """
        for dial_id, turn_id, pred, label in zip(
            batch["dial_ids"], batch["turn_ids"], preds, labels
        ):
            if turn_id not in results[dial_id]:
                results[dial_id][turn_id] = {}
            if pred not in results[dial_id][turn_id]:
                results[dial_id][turn_id] = {"preds": [], "labels": []}

            for i, (p, la) in enumerate(zip(pred, label)):
                if p == 1:
                    results[dial_id][turn_id]["preds"].append(ACT_ONTOLOGY[i])
                if la == 1:
                    results[dial_id][turn_id]["labels"].append(ACT_ONTOLOGY[i])

    def evaluation(
        self, dataloader: DataLoader, epoch: Optional[int] = None, mode: str = "dev"
    ) -> Tuple[float, dict]:
        """Evaluation on dev dataset or tests dataset.
        calculate `accuracy`, `precision`, `recall`, `f1` and `joint_accuracy`

        Args:
            dataloader: see torch.utils.data.DataLoader
            epoch: current training epochs
            mode: train, dev or tests

        Returns:
            specified metric value and formatted prediction results
        """
        self.model.eval()
        desc_prefix = "Evaluating" if mode == "dev" else "Testing"
        eval_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc_prefix)
        all_preds = []
        all_labels = []
        prediction_results = defaultdict(dict)
        with torch.no_grad():
            for step, batch in eval_bar:
                labels, loss, preds = self.eval_forward(batch)
                all_preds.append(preds)
                all_labels.append(labels)
                self.format_results(batch, labels, preds, prediction_results)

                desc = (
                    f"{desc_prefix}： Epoch: {epoch}, "
                    if mode == "dev"
                    else "Best model, "
                )
                desc += f"BCELoss: {loss.item():.3f}"
                eval_bar.set_description(desc)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics_res = eval_metrics(all_preds, all_labels)

        joint_acc = get_joint_acc(prediction_results)
        metrics_res["joint_acc"] = joint_acc

        print("*" * 10 + " Eval Metrics " + "*" * 10)
        print(json.dumps(metrics_res, indent=2))
        return metrics_res[self.config["eval_metric"]], prediction_results

    def train(self) -> None:
        """Training."""
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
                labels = inputs.pop("labels")
                logits = self.model(**inputs)[0]
                loss = self.loss_fct(logits, labels)

                if self.config["n_gpus"] > 1:
                    loss = loss.mean()

                self.model.zero_grad()
                loss.backward()
                clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config["max_grad_norm"]
                )
                self.optimizer.step()

                train_bar.set_description(
                    f"Training： Epoch: {epoch}, Iter: {step}, BCELoss: {loss.item():.3f}"
                )

            eval_metric, _ = self.evaluation(self.eval_dataloader, epoch)
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

        _, prediction_results = self.evaluation(self.test_dataloader, mode="tests")
        dump_json(
            prediction_results,
            os.path.join(self.config["data_path"], "prediction.json"),
        )


def main():
    train_config_name = "policy/bert/train.json"
    common_config_name = "policy/bert/common.json"

    data_urls = {
        "config.json": "http://xbot.bslience.cn/bert-base-chinese/config.json",
        "pytorch_model.bin": "http://xbot.bslience.cn/bert-base-chinese/pytorch_model.bin",
        "vocab.txt": "http://xbot.bslience.cn/bert-base-chinese/vocab.txt",
        "act_ontology.json": "http://xbot.bslience.cn/act_ontology.json",
    }

    train_config = update_config(
        common_config_name, train_config_name, "crosswoz/policy_bert_data"
    )
    train_config["raw_data_path"] = os.path.join(get_data_path(), "crosswoz/raw")

    # download data
    for data_key, url in data_urls.items():
        dst = os.path.join(train_config["data_path"], data_key)
        file_name = data_key.split(".")[0]
        train_config[file_name] = dst
        if not os.path.exists(dst):
            download_from_url(url, dst)

    pl.seed_everything(train_config["seed"])
    trainer = Trainer(train_config)
    trainer.train()
    trainer.eval_test()


if __name__ == "__main__":
    main()
