# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 2:58 下午
# @Author  : zhengjiawei
# @FileName: commands.py
# @Software: PyCharm
import argparse


class BaseXbotCliCommand:
    """
    默认格式
    command_dict = {'train': {'param': [["--data_dir", str, './', "path to dataset."],
                                                 ['--output', str, "./", "path to saved the trained model."],
                                                 ["--batch_size", int, 32, "Batch size for training."],
                                                 ["--learning_rate", float, 3e-5, "Learning rate."],
                                                 ["--device", str, 'cuda:0', "train or test device"],
                                                 ["--seed", int, 2021, "random seed"],
                                                 ["--model", str, "hfl/chinese-bert-wwm-ext",
                                                  "Model's name or path to stored model."]
                                                 ]
                                       },
                             'predict': {'param': [["--data_dir", str, './', "path to dataset."],
                                                   ['--output', str, "./", "path to saved the trained model."],
                                                   ["--batch_size", int, 64, "Batch size for training."],
                                                   ["--learning_rate", float, 3e-9, "Learning rate."],
                                                   ["--device", str, 'cuda:0', "train or test device"],
                                                   ["--seed", int, 2021, "random seed"],
                                                   ["--model", str, "hfl/chinese-bert-wwm-ext",
                                                    "Model's name or path to stored model."]
                                                   ]
                                         }}
    通过param获得subcommand,以双列表的形式存放，每个内层列表中存放4个内容，对应的是参数，参数类型，默认值，参数说明
    """

    def __init__(self, command_dict):
        self.command_dict = command_dict
        self.parser = argparse.ArgumentParser(
            "Xbot CLI tool", usage="xbot cli <command> [<args>]"
        )
        self.commands_parser = self.parser.add_subparsers(
            help="xbot-cli command helpers"
        )

        for key in self.command_dict.keys():
            self.sub_parser = self.commands_parser.add_parser(
                key, help="xbot tool to {} a model on a task.".format(key)
            )
            self.sub_parser.set_defaults()
            self.subcommand_key = self.command_dict.get(key)
            for each in self.subcommand_key.get("param"):
                self.sub_parser.add_argument(
                    each[0], type=each[1], default=each[2], help=each[3]
                )
