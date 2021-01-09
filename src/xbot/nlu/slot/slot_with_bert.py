import os
import json
from typing import Any

from src.xbot.util.nlu_util import NLU
from src.xbot.constants import DEFAULT_MODEL_PATH
from src.xbot.util.path import get_root_path
from src.xbot.util.download import download_from_url
from data.crosswoz.data_process.nlu.nlu_slot_dataloader import Dataloader
from data.crosswoz.data_process.nlu.nlu_slot_postprocess import recover_intent

import torch
from torch import nn
from transformers import BertModel


class SlotWithBert(nn.Module):
    """Slot Extraction with Bert"""

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, model_config, device, slot_dim):
        super(SlotWithBert, self).__init__()
        # label num
        self.slot_num_labels = slot_dim
        self.device = device
        # model
        self.bert = BertModel.from_pretrained(model_config["pretrained_weights"])
        self.dropout = nn.Dropout(model_config["dropout"])
        self.context = model_config["context"]
        self.finetune = model_config["finetune"]
        self.context_grad = model_config["context_grad"]
        self.hidden_units = model_config["hidden_units"]
        # 选择神经元不同的全联接层
        if self.hidden_units > 0:
            if self.context:
                self.slot_classifier = nn.Linear(
                    self.hidden_units, self.slot_num_labels
                )
                self.slot_hidden = nn.Linear(
                    2 * self.bert.config.hidden_size, self.hidden_units
                )
            else:
                self.slot_classifier = nn.Linear(
                    self.hidden_units, self.slot_num_labels
                )
                self.slot_hidden = nn.Linear(
                    self.bert.config.hidden_size, self.hidden_units
                )
            # 初始化参数，服从均匀分布
            nn.init.xavier_uniform_(self.slot_hidden.weight)
        else:
            if self.context:
                self.slot_classifier = nn.Linear(
                    2 * self.bert.config.hidden_size, self.slot_num_labels
                )
            else:
                self.slot_classifier = nn.Linear(
                    self.bert.config.hidden_size, self.slot_num_labels
                )
        nn.init.xavier_uniform_(self.slot_classifier.weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        word_seq_tensor,
        word_mask_tensor,
        tag_seq_tensor=None,
        tag_mask_tensor=None,
        context_seq_tensor=None,
        context_mask_tensor=None,
    ):
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=word_seq_tensor, attention_mask=word_mask_tensor
                )
        else:
            outputs = self.bert(
                input_ids=word_seq_tensor, attention_mask=word_mask_tensor
            )

        # 输入的是word_seq_tensor，bert的输出有两部分，
        # 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
        sequence_output = outputs[0]
        # 这个输出 是获取句子的output
        pooled_output = outputs[1]

        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(
                        input_ids=context_seq_tensor, attention_mask=context_mask_tensor
                    )[1]
            else:
                # 输入context_seq_tensor，同样得到的输出有两个，取第二个
                context_output = self.bert(
                    input_ids=context_seq_tensor, attention_mask=context_mask_tensor
                )[1]
            sequence_output = torch.cat(
                [
                    context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                    sequence_output,
                ],
                dim=-1,
            )
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(
                self.slot_hidden(self.dropout(sequence_output))
            )

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)
        pooled_output = self.dropout(pooled_output)
        outputs = outputs
        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[
                active_tag_loss
            ]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)
            outputs = outputs + (slot_loss,)
        return outputs


class SlotWithBertPredictor(NLU):
    """NLU slot Extraction with Bert 预测器"""

    default_model_config = "nlu/crosswoz_all_context_nlu_slot.json"
    default_model_name = "pytorch-slot-with-bert.pt"
    default_model_url = "http://xbot.bslience.cn/pytorch-slot-with-bert.pt"

    def __init__(self):
        # path
        root_path = get_root_path()
        config_file = os.path.join(
            root_path,
            "src/xbot/config/{}".format(SlotWithBertPredictor.default_model_config),
        )

        # load config
        config = json.load(open(config_file))
        data_path = os.path.join(root_path, config["data_dir"])
        device = config["DEVICE"]

        # load intent, tag vocabulary and dataloader
        intent_vocab = json.load(
            open(os.path.join(data_path, "intent_vocab.json"), encoding="utf-8")
        )
        tag_vocab = json.load(
            open(os.path.join(data_path, "tag_vocab.json"), encoding="utf-8")
        )
        dataloader = Dataloader(
            tag_vocab=tag_vocab,
            intent_vocab=intent_vocab,
            pretrained_weights=config["model"]["pretrained_weights"],
        )
        # load best model
        best_model_path = os.path.join(
            DEFAULT_MODEL_PATH, SlotWithBertPredictor.default_model_name
        )
        if not os.path.exists(best_model_path):
            download_from_url(SlotWithBertPredictor.default_model_url, best_model_path)
        model = SlotWithBert(config["model"], device, dataloader.tag_dim)
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        DEFAULT_MODEL_PATH, SlotWithBertPredictor.default_model_name
                    ),
                    map_location="cpu",
                )
            )
        except Exception as e:
            print(e)
        model.to(device)

        self.model = model
        self.dataloader = dataloader
        print(f"{best_model_path} loaded - {best_model_path}")

    def predict(self, utterance, context=list()):
        # utterance
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        # tag
        ori_tag_seq = ["O"] * len(ori_word_seq)
        context_seq = self.dataloader.tokenizer.encode(
            "[CLS] " + " [SEP] ".join(context[-3:])
        )
        da = {}
        word_seq, tag_seq, new2ori = ori_word_seq, ori_tag_seq, None
        batch_data = [
            [
                ori_word_seq,
                ori_tag_seq,
                da,
                context_seq,
                new2ori,
                word_seq,
                self.dataloader.seq_tag2id(tag_seq),
            ]
        ]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to("cpu") for t in pad_batch)
        (
            word_seq_tensor,
            tag_seq_tensor,
            word_mask_tensor,
            tag_mask_tensor,
            context_seq_tensor,
            context_mask_tensor,
        ) = pad_batch
        # inference
        slot_logits, batch_slot_loss = self.model(
            word_seq_tensor,
            word_mask_tensor,
            tag_seq_tensor,
            tag_mask_tensor,
            context_seq_tensor,
            context_mask_tensor,
        )
        # postprocess
        predicts = recover_intent(
            self.dataloader,
            slot_logits[0],
            tag_mask_tensor[0],
            batch_data[0][0],
            batch_data[0][1],
        )
        return predicts


if __name__ == "__main__":
    slot = SlotWithBertPredictor()
    print(
        slot.predict(
            utterance="北京布提克精品酒店酒店是什么类型，有健身房吗？",
            context=["你好，给我推荐一个评分是5分，价格在100-200元的酒店。", "推荐您去北京布提克精品酒店。"],
        )
    )
