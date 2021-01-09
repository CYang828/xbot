import sys
from typing import Any, Text, NoReturn


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def wrap_with_color(*args: Any, color: Text) -> Text:
    return color + " ".join(str(s) for s in args) + bcolors.ENDC


def print_color(*args: Any, color: Text) -> None:
    output = wrap_with_color(*args, color=color)
    try:
        # colorama is used to fix a regression where colors can not be printed on
        # windows. https://github.com/RasaHQ/rasa/issues/7053
        from colorama import AnsiToWin32

        stream = AnsiToWin32(sys.stdout).stream
        print(output, file=stream)
    except ImportError:
        print(output)


def print_success(*args: Any):
    print_color(*args, color=bcolors.OKGREEN)


def print_info(*args: Any):
    print_color(*args, color=bcolors.OKBLUE)


def print_warning(*args: Any):
    print_color(*args, color=bcolors.WARNING)


def print_error(*args: Any):
    print_color(*args, color=bcolors.FAIL)


def print_error_and_exit(message: Text, exit_code: int = 1) -> NoReturn:
    """Print error message and exit the application."""

    print_error(message)
    sys.exit(exit_code)
