import os
import json

import torch
import torch.nn as nn

from xbot.util.policy_util import Policy
from xbot.util.download import download_from_url
from xbot.util.path import get_config_path, get_data_path
from data.crosswoz.data_process.policy.mle_preprocess import CrossWozVector


class MultiDiscretePolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super(MultiDiscretePolicy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(s_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, a_dim),
        )

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :param sample
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)

        # [a_dim] => [a_dim, 2]
        a_probs = a_probs.unsqueeze(1)
        a_probs = torch.cat([1 - a_probs, a_probs], 1)

        # [a_dim, 2] => [a_dim]
        a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)

        return a

    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)

        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(-1)
        a_probs = torch.cat([1 - a_probs, a_probs], -1)

        # [b, a_dim, 2] => [b, a_dim]
        trg_a_probs = a_probs.gather(-1, a.unsqueeze(-1).long()).squeeze(-1)
        log_prob = torch.log(trg_a_probs)

        return log_prob.sum(-1, keepdim=True)


class MLEPolicy(Policy):
    model_config_name = "policy/mle/inference.json"
    common_config_name = "policy/mle/common.json"

    model_urls = {
        "model.pth": "",
        "sys_da_voc.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/usr_da_voc.json",
        "usr_da_voc.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/usr_da_voc.json",
    }

    def __init__(self):
        super(MLEPolicy, self).__init__()
        # load config
        common_config_path = os.path.join(
            get_config_path(), MLEPolicy.common_config_name
        )
        common_config = json.load(open(common_config_path))
        model_config_path = os.path.join(get_config_path(), MLEPolicy.model_config_name)
        model_config = json.load(open(model_config_path))
        model_config.update(common_config)
        self.model_config = model_config
        self.model_config["data_path"] = os.path.join(
            get_data_path(), "crosswoz/policy_mle_data"
        )
        self.model_config["n_gpus"] = (
            0 if self.model_config["device"] == "cpu" else torch.cuda.device_count()
        )
        self.model_config["device"] = torch.device(self.model_config["device"])

        # download data
        for model_key, url in MLEPolicy.model_urls.items():
            dst = os.path.join(self.model_config["data_path"], model_key)
            file_name = (
                model_key.split(".")[0]
                if not model_key.endswith("pth")
                else "trained_model_path"
            )
            self.model_config[file_name] = dst
            if not os.path.exists(dst) or not self.model_config["use_cache"]:
                download_from_url(url, dst)

        self.vector = CrossWozVector(
            sys_da_voc_json=self.model_config["sys_da_voc"],
            usr_da_voc_json=self.model_config["usr_da_voc"],
        )

        policy = MultiDiscretePolicy(
            self.vector.state_dim, model_config["hidden_size"], self.vector.sys_da_dim
        )

        policy.load_state_dict(torch.load(self.model_config["trained_model_path"]))

        self.policy = policy.to(self.model_config["device"]).eval()
        print(f'>>> {self.model_config["trained_model_path"]} loaded ...')

    def init_session(self):
        pass

    def predict(self, state):
        s_vec = torch.tensor(
            self.vector.state_vectorize(state), device=self.model_config["device"]
        )
        a = self.policy.select_action(s_vec, sample=False).cpu().numpy()
        action = self.vector.action_devectorize(a)
        state["system_action"] = action
        return action
