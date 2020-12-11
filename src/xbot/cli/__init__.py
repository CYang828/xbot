import click

from .. import __version__


@click.command()
@click.option("--count", default=1, help="Number of greetings.")
@click.version_option(version=__version__)
def main(count):
    print(count)
    click.echo("Hello, Xbot!")
