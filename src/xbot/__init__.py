from pathlib import Path
from typing import Text
from importlib.metadata import version, PackageNotFoundError

from .builder import Builder
from .cli.commands import BaseXbotCliCommand


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__ = ["XBot"]


class XBot(BaseXbotCliCommand):
    """Converstaional AI XBot object"""

    def __init__(
        self,
        name: Text = "XBot",
        config_path: Path = "config.yml",
        model_dir: Path = "model/",
        data_dir: Path = "data/",
    ) -> None:

        self.name = name
        self.cli = None
        self.interpreter = None
        self.trainer = None
        self.builder = Builder(config_path, model_dir, data_dir)
        self.command_dict = {}

    def __str__(self):
        return f"<XBot {self.name}>"

    def build(self):
        """call builder to build XBot interpreter and trainer"""
        pass

    def init(self):
        pass

    def interactive(self):
        """interactive with your bot"""
        pass

    def train(self):
        """call trainer to train XBot model"""
        pass
