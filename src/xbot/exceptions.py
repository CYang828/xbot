"""XBot System Level Exception Definition"""


class XBotBaseException(Exception):
    """Base exception class for all errors raised by XBot Open Source."""


class XBotIOException(XBotBaseException):
    """Raised when IO error"""


class XBotConfigException(XBotBaseException):
    """Raised when config error"""


class XBotComponentException(XBotBaseException):
    """Raised when component error"""
