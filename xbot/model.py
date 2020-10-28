import abc


class Model(abc.ABC):
    """XBot Model"""
    def __init__(self):
        self._default_weight = None

    @abc.abstractmethod
    def fit(self, x, y):
        raise NotImplemented
