# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 2:58 下午
# @Author  : zhengjiawei
# @FileName: commands.py
# @Software: PyCharm
import argparse


class BaseXbotCliCommand:
    def __init__(self, *args) -> object:
        pass

    @staticmethod
    def register_subcommand(parser, command_dict: dict) -> object:
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
         command_dict = {'param': [["--data_dir", str, './', "path to dataset."],
                              ['--output', str, "./", "path to saved the trained model."],
                              ["--batch_size", int, 32, "Batch size for training."],
                              ["--learning_rate", float, 3e-5, "Learning rate."],
                              ["--device", str, 'cuda:0', "train or test device"],
                              ["--seed", int, 2021, "random seed"],
                              ["--model", str, "hfl/chinese-bert-wwm-ext",
                               "Model's name or path to stored model."]
                              ]
                    }
                    :param commands_parser:
                    :param command_dict:
        """
        sub_parser = parser.add_parser(
            list(command_dict.keys())[0],
            help="xbot tool to {} a model on a task.".format(
                list(command_dict.keys())[0]
            ),
        )
        for each in command_dict.get(list(command_dict.keys())[0]):
            sub_parser.add_argument(
                each[0], type=each[1], default=each[2], help=each[3]
            )
        sub_parser.set_defaults()
        return sub_parser
