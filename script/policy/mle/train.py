import os
import json
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn

from xbot.util.download import download_from_url
from xbot.dm.policy.mle_policy.mle import MultiDiscretePolicy
from script.policy.mle.utils import DataPreprocessor, f1, set_seed
from xbot.util.path import get_data_path, get_root_path, get_config_path
from data.crosswoz.data_process.policy.mle_preprocess import CrossWozVector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer:
    def __init__(self, config):
        vector = CrossWozVector(
            sys_da_voc_json=config["sys_da_voc"], usr_da_voc_json=config["usr_da_voc"]
        )

        data_preprocessor = DataPreprocessor(config, vector)
        self.data_train = data_preprocessor.create_dataset(
            "train", config["batch_size"]
        )
        self.data_valid = data_preprocessor.create_dataset("val", config["batch_size"])
        self.data_test = data_preprocessor.create_dataset("tests", config["batch_size"])

        self.save_dir = config["output_dir"]
        self.print_per_batch = config["print_per_batch"]
        self.device = config["device"]

        self.policy = MultiDiscretePolicy(
            vector.state_dim, config["hidden_size"], vector.sys_da_dim
        )
        model_path = config["model_path"]
        if model_path:
            print(f"Model {model_path} loaded")
            trained_model_params = torch.load(model_path)
            self.policy.load_state_dict(trained_model_params)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["learning_rate"]
        )

        self.policy.to(self.device)
        if config["n_gpus"] > 0:
            self.policy = nn.DataParallel(self.policy)

        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()

    def policy_loop(self, data):
        s, target_a = (item.to(self.device) for item in data)
        a_weights = self.policy(s)
        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a

    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.0
        pbar = tqdm(enumerate(self.data_train), total=len(self.data_train))
        for i, data in pbar:
            self.optimizer.zero_grad()
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            self.optimizer.step()

            pbar.set_description(
                f"Epoch {epoch}, iter {i}, loss_a: {a_loss / (i + 1):.3f}"
            )

    def imit_eval(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """
        self.policy.eval()
        a_loss = 0.0
        pbar = tqdm(enumerate(self.data_valid), total=len(self.data_valid))
        for i, data in pbar:
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_valid)
        print(f"Validation, epoch {epoch}, loss_a: {a_loss:.3f}")
        if a_loss < best:
            print("Best model saved")
            best = a_loss
            self.save(self.save_dir, f"Epoch-{epoch:d}-Loss-{best:.4f}")

        a_loss = 0.0
        pbar = tqdm(enumerate(self.data_test), total=len(self.data_test))
        for i, data in pbar:
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_test)
        print(f"Test, epoch {epoch}, loss_a: {a_loss:.3f}")
        return best

    def calc_metrics(self):
        self.policy.eval()
        a_tp, a_fp, a_fn = 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.data_test):
                s, target_a = (item.to(self.device) for item in data)
                a_weights = self.policy(s)
                a = a_weights.ge(0)
                # TODO: fix batch ，这个不是 batch f1，batch f1 要求每个样本里的所有 action 都正确，才算正确
                tp, fp, fn = f1(a, target_a)
                a_tp += tp
                a_fp += fp
                a_fn += fn

        precision = a_tp / (a_tp + a_fp)
        recall = a_tp / (a_tp + a_fn)
        f1_score = 2 * precision * recall / (precision + recall)
        print(a_tp, a_fp, a_fn, f1_score)

    def save(self, directory, metric):
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_save_path = os.path.join(directory, f"mle-{metric}")
        model_to_save = (
            self.policy.module if hasattr(self.policy, "module") else self.policy
        )
        model_to_save = deepcopy(model_to_save)
        torch.save(model_to_save.cpu().state_dict(), model_save_path)


def main():
    model_config_name = "policy/mle/train.json"
    common_config_name = "policy/mle/common.json"

    data_urls = {
        "sys_da_voc.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/usr_da_voc.json",
        "usr_da_voc.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/usr_da_voc.json",
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

    model_config["data_path"] = os.path.join(
        get_data_path(), "crosswoz/policy_mle_data"
    )
    model_config["raw_data_path"] = os.path.join(get_data_path(), "crosswoz/raw")
    model_config["output_dir"] = os.path.join(root_path, model_config["output_dir"])
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
        file_name = data_key.split(".")[0]
        model_config[file_name] = dst
        if not os.path.exists(dst):
            download_from_url(url, dst)

    print(f">>> Train configs:")
    print("\t", model_config)

    set_seed(model_config["random_seed"])

    agent = Trainer(model_config)

    # 训练
    if model_config["do_train"]:
        start_epoch = (
            0
            if not model_config["model_path"]
            else int(model_config["model_path"].split("-")[2]) + 1
        )
        best = float("inf")
        for epoch in tqdm(range(start_epoch, model_config["num_epochs"]), desc="Epoch"):
            agent.imitating(epoch)
            best = agent.imit_eval(epoch, best)

    agent.calc_metrics()


if __name__ == "__main__":
    main()
