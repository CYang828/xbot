from pathlib import Path

from .util.config import read_yaml_file


class Builder(object):
    """
    build interpreter and trainer based on config file,
    component object manager for reusing,
    check config file and generage readable help for user.

    :return :
        interpreter
        trainer
    """

    def __init__(
        self,
        config_path: Path = "config.yml",
        model_dir: Path = "model/",
        data_dir: Path = "data/",
    ) -> None:

        self.config_path = config_path
        self.model_dir = model_dir
        self.data_dir = data_dir

    def from_config(self):
        read_yaml_file(self.config_path)

    def from_yaml(self, yaml_path: Path):
        pass
