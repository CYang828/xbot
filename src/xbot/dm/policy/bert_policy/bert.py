import os
import random
from typing import List, Dict, Tuple

import torch

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

from xbot.util.db_query import Database
from xbot.util.policy_util import Policy
from xbot.util.file_util import load_json
from xbot.util.path import get_config_path
from xbot.util.download import download_from_url
from data.crosswoz.data_process.policy.bert_proprecess import str2id, pad


class BertPolicy(Policy):
    inference_config_name = "policy/bert/inference.json"
    common_config_name = "policy/bert/common.json"

    data_urls = {
        "config.json": "http://xbot.bslience.cn/bert-policy/config.json",
        "pytorch_model.bin": "http://xbot.bslience.cn/bert-policy/pytorch_model.bin",
        "vocab.txt": "http://xbot.bslience.cn/bert-policy/vocab.txt",
        "act_ontology.json": "http://xbot.bslience.cn/bert-policy/act_ontology.json",
    }

    def __init__(self):
        super(BertPolicy, self).__init__()
        # load config
        infer_config = self.load_config()

        # download data
        model_dir = os.path.join(infer_config["data_path"], "trained_model")
        # model_dir = os.path.join('/xhp/xbot/output/policy/bert', 'Epoch-19-f1-0.903')
        infer_config["model_dir"] = model_dir
        self.download_data(infer_config, model_dir)
        # 应该保持和训练使用的一致，否则 label 顺序不一致，TODO 训练时对 act_ontology 排序
        self.act_ontology = load_json(infer_config["act_ontology"])
        self.num_act = len(self.act_ontology)

        model_config = BertConfig.from_pretrained(infer_config["model_dir"])
        model_config.num_labels = self.num_act
        self.model = BertForSequenceClassification.from_pretrained(
            infer_config["model_dir"], config=model_config
        )
        self.tokenizer = BertTokenizer.from_pretrained(infer_config["model_dir"])

        self.model.eval()
        self.model.to(infer_config["device"])

        self.db = Database()
        self.config = infer_config
        self.threshold = infer_config["threshold"]

    @staticmethod
    def download_data(infer_config: dict, model_dir: str) -> None:
        """Download trained model for inference.

        Args:
            infer_config: config used for inference
            model_dir: model save directory
        """
        for data_key, url in BertPolicy.data_urls.items():
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            dst = os.path.join(model_dir, data_key)
            file_name = data_key.split(".")[0]
            infer_config[file_name] = dst
            if not os.path.exists(dst):
                download_from_url(url, dst)

    @staticmethod
    def load_config() -> dict:
        """Load config for inference.

        Returns:
            config dict
        """
        common_config_path = os.path.join(
            get_config_path(), BertPolicy.common_config_name
        )
        infer_config_path = os.path.join(
            get_config_path(), BertPolicy.inference_config_name
        )
        common_config = load_json(common_config_path)
        infer_config = load_json(infer_config_path)
        infer_config.update(common_config)
        infer_config["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        infer_config["data_path"] = os.path.join(
            get_data_path(), "crosswoz/policy_bert_data"
        )
        if not os.path.exists(infer_config["data_path"]):
            os.makedirs(infer_config["data_path"])
        return infer_config

    def preprocess(
        self, belief_state: Dict[str, dict], cur_domain: str, history: List[tuple]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess raw dialogue data to bert inputs.

        Args:
            belief_state: see `xbot/util/state.py`
            cur_domain: current domain
            history: dialogue history, [('usr', 'xxx'), ('sys', 'xxx'), ...]

        Returns:
            bert inputs, contain input_ids, token_type_ids, attention_mask
        """
        sys_utter = "对话开始"
        usr_utter = "对话开始"
        if len(history) > 0:
            usr_utter = history[-1][1]
        if len(history) > 2:
            sys_utter = history[-2][1]

        source = self.get_source(belief_state, cur_domain)
        input_ids, token_type_ids = str2id(self.tokenizer, sys_utter, usr_utter, source)
        attention_mask, input_ids, token_type_ids = pad([input_ids], [token_type_ids])

        return input_ids, token_type_ids, attention_mask

    @staticmethod
    def get_source(belief_state: Dict[str, dict], cur_domain: str) -> str:
        """Take constraints in belief state.
        TODO: belief_state 要不要换成 sys_state

        Args:
            belief_state: current belief state
            cur_domain: current domain

        Returns:
            concatenate all slot-value pair
        """
        if cur_domain is None:
            return "无结果"
        source = []
        for slot, value in belief_state[cur_domain].items():
            if not value:
                continue
            source.append(slot + "是" + value)
        source = "，".join(source)
        if not source:
            source = "无结果"
        return source

    def predict(self, state: dict) -> List[list]:
        """Predict the next actions of system.

        Args:
            state: current system state

        Returns:
            a list of actions of system will take
        """
        belief_state = state["belief_state"]
        cur_domain = state["cur_domain"]
        history = state["history"]
        db_res = self.db.query(belief_state, cur_domain)
        # 数据库中查询出来的结果中同时包括了起点和终点，所以不能随机选择一个
        if cur_domain != "地铁" and db_res:
            db_res = random.choice(db_res)[1]

        preds = self.forward(belief_state, cur_domain, history)

        sys_das = []
        for i, pred in enumerate(preds):
            if pred == 1:
                act = self.act_ontology[i]
                if "酒店设施" in act:
                    domain, intent, slot, facility = act.split("-")
                    value = (
                        "是"
                        if db_res and "酒店设施" in db_res and facility in db_res["酒店设施"]
                        else "否"
                    )
                    sys_das.append([intent, domain, slot + "-" + facility, value])
                    continue
                domain, intent, slot = act.split("-")
                if intent == "General":
                    sys_das.append([intent, domain, "none", "none"])
                if domain == cur_domain and db_res:
                    self.get_sys_das(db_res, domain, intent, slot, sys_das)
        return sys_das

    def forward(
        self, belief_state: Dict[str, dict], cur_domain: str, history: List[tuple]
    ) -> torch.Tensor:
        """Forward step, get predictions.

        Args:
            belief_state: see `xbot/util/state.py`
            cur_domain: current domain
            history: dialogue history, [('usr', 'xxx'), ('sys', 'xxx'), ...]

        Returns:
            model predictions
        """
        input_ids, token_type_ids, attention_mask = [
            item.to(self.config["device"])
            for item in self.preprocess(belief_state, cur_domain, history)
        ]
        logits = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[0]
        probs = torch.sigmoid(logits)[0]
        preds = (probs > self.threshold).float()
        return preds

    def get_sys_das(
        self, db_res: dict, domain: str, intent: str, slot: str, sys_das: list
    ) -> None:
        """Construct system actions according to different domains and values taken from database.

        Args:
            db_res: database query results
            domain: current domain
            intent: system's intent, such as: Inform, Recommend,...
            slot: candidate slot, such ad: '价格', '名称', ...
            sys_das: system actions are saved into sys_das
        """
        if domain == "地铁":
            if slot == "出发地附近地铁站":
                value = self.get_metro_das(db_res, "起点")
                sys_das.append([intent, domain, slot, value])
            else:
                value = self.get_metro_das(db_res, "终点")
                sys_das.append([intent, domain, slot, value])
        elif domain == "出租":
            if slot == "车型":
                sys_das.append([intent, domain, slot, "#CX"])
            elif slot == "车牌":
                sys_das.append([intent, domain, slot, "#CP"])
        else:
            value = db_res.get(slot, "无")
            if isinstance(value, list):
                if not value:
                    value = "无"
                else:
                    value = random.choice(value)
            sys_das.append([intent, domain, slot, value])

    @staticmethod
    def get_metro_das(db_res: dict, slot: str) -> str:
        """Take departure and destination of metro from database query results.

        Args:
            db_res: database query results
            slot: '起点' or '终点'

        Returns:
            specified departure or destination
        """
        metros = [res for res in db_res if slot in res[0]]
        value = "无"
        if not metros:
            return value
        for metro in metros:
            value = metro[1]["地铁"]
            if value is not None:
                break
        return value


if __name__ == "__main__":
    from xbot.dm.dst.rule_dst.rule import RuleDST
    from xbot.util.path import get_data_path
    from xbot.util.file_util import read_zipped_json
    from script.policy.rule.rule_test import eval_metrics
    from tqdm import tqdm

    rule_dst = RuleDST()
    bert_policy = BertPolicy()
    train_path = os.path.join(get_data_path(), "crosswoz/raw/train.json.zip")
    train_examples = read_zipped_json(train_path, "train.json")

    sys_state_action_pairs = {}
    for id_, dialogue in tqdm(train_examples.items()):
        sys_state_action_pair = {}
        sess = dialogue["messages"]
        rule_dst.init_session()
        for i, turn in enumerate(sess):
            if turn["role"] == "usr":
                rule_dst.update(usr_da=turn["dialog_act"])
                rule_dst.state["user_action"].clear()
                rule_dst.state["user_action"].extend(turn["dialog_act"])
                rule_dst.state["history"].append(["usr", turn["content"]])
                if i + 2 == len(sess):
                    rule_dst.state["terminated"] = True
            else:
                rule_dst.state["history"].append(["sys", turn["content"]])
                for domain, svs in turn["sys_state"].items():
                    for slot, value in svs.items():
                        if slot != "selectedResults":
                            rule_dst.state["belief_state"][domain][slot] = value

                pred_sys_act = bert_policy.predict(rule_dst.state)
                sys_state_action_pair[str(i)] = {
                    "gold_sys_act": [tuple(act) for act in turn["dialog_act"]],
                    "pred_sys_act": [tuple(act) for act in pred_sys_act],
                }
                rule_dst.state["system_action"].clear()
                rule_dst.state["system_action"].extend(turn["dialog_act"])

        sys_state_action_pairs[id_] = sys_state_action_pair

    f1, precision, recall, joint_acc = eval_metrics(sys_state_action_pairs)
    print(
        f"f1: {f1:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, joint_acc: {joint_acc:.3f}"
    )
    # f1: 0.499, precision: 0.529, recall: 0.472, joint_acc: 0.372
