from abc import ABC

from src.xbot.constants import DEFAULT_MODEL_PATH
from src.xbot.util.download import download_from_url


class Model(ABC):
    """XBot Model"""

    def __init__(self):
        super(Model, self).__init__()

    @staticmethod
    def load_from_net(url):
        download_from_url(url, DEFAULT_MODEL_PATH)
