import argparse
from src.xbot.cli.commands import BaseXbotCliCommand


# class MyTestCase(BaseXbotCliCommand):
#     def __init__(self):
#         super().__init__(command_dict)
#         self.sub_parser.add_argument('--version', type=str, default='0.0.1', help='xbot version')
#
#
# command_dict = {'train': {'param': [["--data_dir", str, './', "path to dataset."],
#                                     ['--output', str, "./", "path to saved the trained model."],
#                                     ["--batch_size", int, 32, "Batch size for training."],
#                                     ["--learning_rate", float, 3e-5, "Learning rate."],
#                                     ["--device", str, 'cuda:0', "train or test device"],
#                                     ["--seed", int, 2021, "random seed"],
#                                     ["--model", str, "hfl/chinese-bert-wwm-ext",
#                                      "Model's name or path to stored model."]
#                                     ]
#                           },
#                 'predict': {'param': [["--data_dir", str, './', "path to dataset."],
#                                       ['--output', str, "./", "path to saved the trained model."],
#                                       ["--batch_size", int, 64, "Batch size for training."],
#                                       ["--learning_rate", float, 3e-9, "Learning rate."],
#                                       ["--device", str, 'cuda:0', "train or test device"],
#                                       ["--seed", int, 2021, "random seed"],
#                                       ["--model", str, "hfl/chinese-bert-wwm-ext",
#                                        "Model's name or path to stored model."]
#                                       ]
#                             }}
#
# def test_main_succeeds():
#     subcommand = MyTestCase()
#     result = subcommand.parser.parse_args(['predict', '--version', '0.0.2'])
#     assert result.version == '0.0.2'


class MyTestCase1(BaseXbotCliCommand):
    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def train():
        print("正确运行")


class MyTestCase2(BaseXbotCliCommand):
    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def train():
        print("正确运行")


class MyTestCase3(BaseXbotCliCommand):
    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def train():
        print("正确运行")


def test_main_succeeds():
    command_dict1 = {
        "train": [
            ["--data_dir", str, "./", "path to dataset."],
            ["--output", str, "./", "path to saved the trained model."],
            ["--batch_size", int, 32, "Batch size for training."],
            ["--learning_rate", float, 3e-5, "Learning rate."],
            ["--device", str, "cuda:0", "train or test device"],
            ["--seed", int, 2021, "random seed"],
            [
                "--model",
                str,
                "hfl/chinese-bert-wwm-ext",
                "Model's name or path to stored model.",
            ],
        ]
    }

    command_dict2 = {
        "test": [
            ["--data_dir", str, "./", "path to dataset."],
            ["--output", str, "./", "path to saved the trained model."],
            ["--batch_size", int, 32, "Batch size for training."],
            ["--learning_rate", float, 3e-5, "Learning rate."],
            ["--device", str, "cuda:0", "train or test device"],
            ["--seed", int, 2021, "random seed"],
            [
                "--model",
                str,
                "hfl/chinese-bert-wwm-ext",
                "Model's name or path to stored model.",
            ],
        ]
    }

    command_dict3 = {
        "predict": [
            ["--data_dir", str, "./", "path to dataset."],
            ["--output", str, "./", "path to saved the trained model."],
            ["--batch_size", int, 32, "Batch size for training."],
            ["--learning_rate", float, 3e-5, "Learning rate."],
            ["--device", str, "cuda:0", "train or test device"],
            ["--seed", int, 2021, "random seed"],
            [
                "--model",
                str,
                "hfl/chinese-bert-wwm-ext",
                "Model's name or path to stored model.",
            ],
        ]
    }
    parser = argparse.ArgumentParser(
        "Xbot CLI tool", usage="xbot cli <command> [<args>]"
    )
    commands_parser = parser.add_subparsers(help="xbot-cli command helpers")
    MyTestCase1.register_subcommand(commands_parser, command_dict1)
    MyTestCase2.register_subcommand(commands_parser, command_dict2)
    MyTestCase3.register_subcommand(commands_parser, command_dict3)
    args = parser.parse_args(["predict", "--batch_size", "12"])
    assert args.batch_size == 12
