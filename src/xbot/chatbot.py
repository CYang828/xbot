from .exceptions import XBotBaseException


class ChatbotBaseException(XBotBaseException):
    pass


class Chatbot(object):
    """对话机器人类"""

    def __init__(self, name):
        self._name = name if name else "XBot"
        self._history = []

    def history(self):
        pass

    def get_latest_response(self):
        pass

    def response_from(self, sequence, **kwargs):
        pass
