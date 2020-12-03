from abc import ABC

from xbot.constants import DEFAULT_MODEL_PATH
from xbot.util.download import download_from_url

import pytorch_lightning as pl


class Model(pl.LightningDataModule, ABC):
    """XBot Model"""
    def __init__(self):
        super(Model, self).__init__()

    @staticmethod
    def load_from_net(url):
        download_from_url(url, DEFAULT_MODEL_PATH)

