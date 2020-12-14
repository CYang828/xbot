"""Natural Language Generation Interface"""
import os
import sys

root_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
sys.path.append(root_dir)
from .module import Module


class NLG(Module):
    """Base class for NLG model."""

    def generate(self, action):
        """Generate a natural language utterance conditioned on the dialog act.

        Args:
            action (list of list):
                The dialog action produced by dialog policy module, which is in dialog act format.
        Returns:
            utterance (str):
                A natural langauge utterance.
        """
        return ""
