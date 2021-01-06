from src.xbot.cli.commands import register_subcommand

command_dict_0 = {'train': [["--data_dir", str, './', "path to dataset."],
                            ['--output', str, "./", "path to saved the trained model."],
                            ["--batch_size", int, 32, "Batch size for training."],
                            ["--learning_rate", float, 3e-5, "Learning rate."],
                            ["--device", str, 'cuda:0', "train or test device"],
                            ["--seed", int, 2021, "random seed"],
                            ["--model", str, "hfl/chinese-bert-wwm-ext",
                             "Model's name or path to stored model."]
                            ]
    ,
                  'predict': [["--data_dir", str, './', "path to dataset."],
                              ['--output', str, "./", "path to saved the trained model."],
                              ["--batch_size", int, 64, "Batch size for training."],
                              ["--learning_rate", float, 3e-9, "Learning rate."],
                              ["--device", str, 'cuda:0', "train or test device"],
                              ["--seed", int, 2021, "random seed"],
                              ["--model", str, "hfl/chinese-bert-wwm-ext",
                               "Model's name or path to stored model."]
                              ]
                  }


class MyTestCase2:

    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    @register_subcommand()
    def train(command_dict2, list_) -> object:
        print(list_)

    @staticmethod
    @register_subcommand()
    def predict(command_dict1) -> object:
        pass


def test_main_succeeds():
    # list_ = [1, 2, 3]
    # parse = MyTestCase2.train(command_dict_0, list_)
    parse1 = MyTestCase2.predict(command_dict_0)
    assert parse1.seed == 2022
