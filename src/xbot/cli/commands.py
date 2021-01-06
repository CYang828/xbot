# -*- coding: utf-8 -*-
# @Time    : 2020/12/25 2:58 下午
# @Author  : zhengjiawei
# @FileName: commands.py
# @Software: PyCharm
import argparse
from functools import wraps


# class BaseXbotCliCommand:
#
#     def __init__(self, *args) -> object:
#         self.parser = argparse.ArgumentParser("Xbot CLI tool", usage="xbot cli <command> [<args>]")
#         self.commands_parser = self.parser.add_subparsers(help="xbot-cli command helpers")


class register_subcommand(object):
    def __call__(self, fn):
        @wraps(fn)
        def wrapper(command_dict, *args, **kwargs):
            self.fn = fn
            fn_name = fn.__name__
            parser = argparse.ArgumentParser("Xbot CLI tool", usage="xbot cli <command> [<args>]")
            commands_parser = parser.add_subparsers(help="xbot-cli command helpers")
            sub_parser = commands_parser.add_parser(fn_name,
                                                    help="xbot tool to {} a model on a task.".format(
                                                        fn_name))

            for each in command_dict.get(fn_name):
                sub_parser.add_argument(each[0], type=each[1], default=each[2], help=each[3])
            sub_parser.set_defaults()
            args_ = parser.parse_args([fn_name, '--seed', '2022'])
            fn(command_dict, *args, **kwargs)
            return args_

        return wrapper
