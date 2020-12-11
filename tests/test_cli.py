import click.testing
import pytest

from xbot import cli


@pytest.fixture
def runner():
    return click.testing.CliRunner()


def test_main_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
