# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 2:58 下午
# @Author  : zhengjiawei
# @FileName: commands.py
# @Software: PyCharm

import argparse


class BaseXbotCLICommand:
    def __init__(self):
        self.command_dict = {'train': {'param': [["--data_dir", str, True, "path to dataset."],
                                                 ['--output', str, "./", "path to saved the trained model."],
                                                 ["--batch_size", int, 32, "Batch size for training."],
                                                 ["--learning_rate", float, 3e-5, "Learning rate."],
                                                 ["--device", str, 'cuda:0', "train or test device"],
                                                 ["--seed", int, 2021, "random seed"],
                                                 ["--model", str, "hfl/chinese-bert-wwm-ext",
                                                  "Model's name or path to stored model."]
                                                 ]

                                       },
                             'predict': {'param': [["--data_dir", str, True, "path to dataset."],
                                                   ['--output', str, "./", "path to saved the trained model."],
                                                   ["--batch_size", int, 64, "Batch size for training."],
                                                   ["--learning_rate", float, 3e-9, "Learning rate."],
                                                   ["--device", str, 'cuda:0', "train or test device"],
                                                   ["--seed", int, 2021, "random seed"],
                                                   ["--model", str, "hfl/chinese-bert-wwm-ext",
                                                    "Model's name or path to stored model."]
                                                   ]

                                         }

                             }
        self.parser = argparse.ArgumentParser("Xbot CLI tool", usage="xbot cli <command> [<args>]")
        self.commands_parser = self.parser.add_subparsers(help="xbot-cli command helpers")
        self.train_parser = self.commands_parser.add_parser("train", help="xbot tool to train a model on a task.")
        self.train_parser.set_defaults()

        self.subcommand_train = self.command_dict.get('train')
        for each in self.subcommand_train.get('param'):
            self.train_parser.add_argument(each[0], type=each[1], default=each[2], help=each[3])
        self.predict_parses = self.commands_parser.add_parser("predict", help="xbot tool to predict a model on a task.")
        self.predict_parses.set_defaults()
        self.subcommand_predict = self.command_dict.get('predict')
        for each in self.subcommand_predict.get('param'):
            self.predict_parses.add_argument(each[0], type=each[1], default=each[2], help=each[3])


class Runcommand(BaseXbotCLICommand):
    def __init__(self):
        super(Runcommand, self).__init__()
        self.predict_parses.add_argument('--version', type=str, default='0.0.1', help='xbot version')



