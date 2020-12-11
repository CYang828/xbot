import json
import os
import time
import logging
import torch

from xbot.util.path import get_root_path, get_config_path, get_data_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logging_handler(log_dir, extra=""):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        "{}/log_{}.txt".format(log_dir, current_time + extra)
    )
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=DEVICE)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=DEVICE)
    return data


def update_config(common_config_name, train_config_name, task_path):
    root_path = get_root_path()
    common_config_path = os.path.join(get_config_path(), common_config_name)
    train_config_path = os.path.join(get_config_path(), train_config_name)
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
    train_config["data_path"] = os.path.join(get_data_path(), task_path)
    train_config["output_dir"] = os.path.join(root_path, train_config["output_dir"])
    if not os.path.exists(train_config["data_path"]):
        os.makedirs(train_config["data_path"])
    if not os.path.exists(train_config["output_dir"]):
        os.makedirs(train_config["output_dir"])
    return train_config
