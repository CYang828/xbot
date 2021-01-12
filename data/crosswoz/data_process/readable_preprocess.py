#!/usr/bin/env python3
import json
import os
import zipfile
import sys
from collections import OrderedDict

from xbot.util.path import get_data_path


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, "r")
    return json.load(archive.open(filename))


def preprocess(mode):
    assert mode == "all" or mode == "usr" or mode == "sys"

    # path
    data_key = ["train", "val", "test"]
    data = {}
    for key in data_key:
        # read crosswoz source data from json.zip
        data[key] = read_zipped_json(
            os.path.join(get_data_path(), "crosswoz/raw", key + ".json.zip"),
            key + ".json",
        )
        print("load {}, size {}".format(key, len(data[key])))

    # generate train, val, tests dataset
    for key in data_key:
        sessions = []
        for no, sess in data[key].items():
            processed_data = OrderedDict()
            processed_data["sys-usr"] = sess["sys-usr"]
            processed_data["type"] = sess["type"]
            processed_data["task description"] = sess["task description"]
            messages = sess["messages"]
            processed_data["turns"] = [
                OrderedDict(
                    {
                        "role": message["role"],
                        "utterance": message["content"],
                        "dialog_act": message["dialog_act"],
                    }
                )
                for message in messages
            ]
            sessions.append(processed_data)
        json.dump(
            sessions,
            open(
                os.path.join(
                    get_data_path(),
                    "crosswoz/readable_data",
                    f"readabe_{key}_data.json",
                ),
                "w",
                encoding="utf-8",
            ),
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        print(os.path.join(
                    get_data_path(),
                    "crosswoz/readable_data",
                    f"readabe_{key}_data.json",
                ))


if __name__ == "__main__":
    # dialogue role: all, usr, sys
    preprocess(sys.argv[1])
