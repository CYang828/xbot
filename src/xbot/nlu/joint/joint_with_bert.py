import os
import json
from typing import Any

from src.xbot.constants import DEFAULT_MODEL_PATH
from src.xbot.util.nlu_util import NLU
from src.xbot.util.path import get_root_path
from src.xbot.util.download import download_from_url
from data.crosswoz.data_process.nlu.nlu_dataloader import Dataloader
from data.crosswoz.data_process.nlu.nlu_postprocess import recover_intent

import torch
from torch import nn
from transformers import BertModel


class JointWithBert(nn.Module):
    """联合NLU模型"""

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, model_config, device, slot_dim, intent_dim, intent_weight=None):
        super(JointWithBert, self).__init__()
        # count of intent and tag
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device

        # init weight
        self.intent_weight = (
            intent_weight
            if intent_weight is not None
            else torch.tensor([1.0] * intent_dim)
        )

        # model
        self.bert = BertModel.from_pretrained(model_config["pretrained_weights"])
        self.dropout = nn.Dropout(model_config["dropout"])
        self.context = model_config["context"]
        self.finetune = model_config["finetune"]
        self.context_grad = model_config["context_grad"]
        self.hidden_units = model_config["hidden_units"]
        if self.hidden_units > 0:
            if self.context:
                self.intent_classifier = nn.Linear(
                    self.hidden_units, self.intent_num_labels
                )
                self.slot_classifier = nn.Linear(
                    self.hidden_units, self.slot_num_labels
                )
                self.intent_hidden = nn.Linear(
                    2 * self.bert.config.hidden_size, self.hidden_units
                )
                self.slot_hidden = nn.Linear(
                    2 * self.bert.config.hidden_size, self.hidden_units
                )
            else:
                self.intent_classifier = nn.Linear(
                    self.hidden_units, self.intent_num_labels
                )
                self.slot_classifier = nn.Linear(
                    self.hidden_units, self.slot_num_labels
                )
                self.intent_hidden = nn.Linear(
                    self.bert.config.hidden_size, self.hidden_units
                )
                self.slot_hidden = nn.Linear(
                    self.bert.config.hidden_size, self.hidden_units
                )
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(
                    2 * self.bert.config.hidden_size, self.intent_num_labels
                )
                self.slot_classifier = nn.Linear(
                    2 * self.bert.config.hidden_size, self.slot_num_labels
                )
            else:
                self.intent_classifier = nn.Linear(
                    self.bert.config.hidden_size, self.intent_num_labels
                )
                self.slot_classifier = nn.Linear(
                    self.bert.config.hidden_size, self.slot_num_labels
                )
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        word_seq_tensor,
        word_mask_tensor,
        tag_seq_tensor=None,
        tag_mask_tensor=None,
        intent_tensor=None,
        context_seq_tensor=None,
        context_mask_tensor=None,
    ):
        # 如果不进行finetune
        if not self.finetune:
            self.bert.eval()
            # 参数不更新
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=word_seq_tensor, attention_mask=word_mask_tensor
                )
        else:
            outputs = self.bert(
                input_ids=word_seq_tensor, attention_mask=word_mask_tensor
            )
        # 获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
        sequence_output = outputs[0]
        # 这个输出是获取句子的output
        pooled_output = outputs[1]

        # 如果有上下文信息
        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(
                        input_ids=context_seq_tensor, attention_mask=context_mask_tensor
                    )[1]
            else:
                # 将上下文信息进行bert训练并获得整个句子的output
                context_output = self.bert(
                    input_ids=context_seq_tensor, attention_mask=context_mask_tensor
                )[1]
                # 将上下文得到输出和word_seq_tensor得到的输出进行拼接
            sequence_output = torch.cat(
                [
                    context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                    sequence_output,
                ],
                dim=-1,
            )
            # 将上下文得到输出和之前获取句子的output进行拼接
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        # 经过dropout、Linear、relu层
        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(
                self.slot_hidden(self.dropout(sequence_output))
            )
            pooled_output = nn.functional.relu(
                self.intent_hidden(self.dropout(pooled_output))
            )
        # 经过dropout
        sequence_output = self.dropout(sequence_output)
        # 经过Linear层
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        outputs = outputs + (intent_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[
                active_tag_loss
            ]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)
            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)
        return outputs


class JointWithBertPredictor(NLU):
    """NLU Joint with Bert 预测器"""

    default_model_config = "nlu/crosswoz_all_context_joint_nlu.json"
    default_model_name = "pytorch-model-nlu-joint.pt"
    default_model_url = "http://xbot.bslience.cn/pytorch-joint-with-bert.pt"

    def __init__(self):
        root_path = get_root_path()
        config_file = os.path.join(
            root_path,
            "src/xbot/config/{}".format(JointWithBertPredictor.default_model_config),
        )
        config = json.load(open(config_file))
        device = config["DEVICE"]
        data_dir = os.path.join(root_path, config["data_dir"])

        intent_vocab = json.load(
            open(os.path.join(data_dir, "intent_vocab.json"), encoding="utf-8")
        )
        tag_vocab = json.load(
            open(os.path.join(data_dir, "tag_vocab.json"), encoding="utf-8")
        )
        dataloader = Dataloader(
            intent_vocab=intent_vocab,
            tag_vocab=tag_vocab,
            pretrained_weights=config["model"]["pretrained_weights"],
        )

        best_model_path = os.path.join(
            DEFAULT_MODEL_PATH, JointWithBertPredictor.default_model_name
        )
        if not os.path.exists(best_model_path):
            download_from_url(JointWithBertPredictor.default_model_url, best_model_path)

        model = JointWithBert(
            config["model"], device, dataloader.tag_dim, dataloader.intent_dim
        )
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        DEFAULT_MODEL_PATH, JointWithBertPredictor.default_model_name
                    ),
                    map_location="cpu",
                )
            )
        except Exception as e:
            print(e)
        model.to(device)

        self.model = model
        self.dataloader = dataloader
        print(f"{best_model_path} loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        ori_tag_seq = ["O"] * len(ori_word_seq)
        context_seq = self.dataloader.tokenizer.encode(
            "[CLS] " + " [SEP] ".join(context[-3:])
        )
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = ori_word_seq, ori_tag_seq, None
        batch_data = [
            [
                ori_word_seq,
                ori_tag_seq,
                intents,
                da,
                context_seq,
                new2ori,
                word_seq,
                self.dataloader.seq_tag2id(tag_seq),
                self.dataloader.seq_intent2id(intents),
            ]
        ]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to("cpu") for t in pad_batch)
        (
            word_seq_tensor,
            tag_seq_tensor,
            intent_tensor,
            word_mask_tensor,
            tag_mask_tensor,
            context_seq_tensor,
            context_mask_tensor,
        ) = pad_batch
        slot_logits, intent_logits = self.model(
            word_seq_tensor,
            word_mask_tensor,
            context_seq_tensor=context_seq_tensor,
            context_mask_tensor=context_mask_tensor,
        )
        intent = recover_intent(
            self.dataloader,
            intent_logits[0],
            slot_logits[0],
            tag_mask_tensor[0],
            batch_data[0][0],
            batch_data[0][-4],
        )
        return intent


if __name__ == "__main__":
    nlu = JointWithBertPredictor()
    print(
        nlu.predict(
            "北京布提克精品酒店酒店是什么类型，有健身房吗？",
            ["你好，给我推荐一个评分是5分，价格在100-200元的酒店。", "推荐您去北京布提克精品酒店。"],
        )
    )
