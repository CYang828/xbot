import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

logging.basicConfig(level=logging.INFO)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


model_args = ModelArguments(
    model_name_or_path="distilbert-base-cased",
)
data_args = DataTrainingArguments(task_name="mnli", data_dir="./glue_data/MNLI")
training_args = TrainingArguments(
    output_dir="./models/model_name",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_gpu_train_batch_size=32,
    per_gpu_eval_batch_size=128,
    num_train_epochs=1,
    logging_steps=500,
    logging_first_step=True,
    save_steps=1000,
    evaluate_during_training=True,
)

set_seed(training_args.seed)

num_labels = glue_tasks_num_labels[data_args.task_name]

config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
)

# Get datasets
train_dataset = GlueDataset(data_args, tokenizer=tokenizer, limit_length=100_000)
eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode='dev')


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics(data_args.task_name, preds, p.label_ids)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
