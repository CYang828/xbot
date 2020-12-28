from src.xbot.cli.commands import BaseXbotCLICommand
import pytest


class MyTestCase(BaseXbotCLICommand):
    def __init__(self):
        super().__init__()
        self.predict_parses.add_argument('--version', type=str, default='0.0.1', help='xbot version')


def test_main_succeeds():
    subcommand = MyTestCase()
    result = subcommand.parser.parse_args(['predict', '--version', '0.0.2'])
    assert result.version == '0.0.2'
