import os
import datetime
from abc import ABC
from typing import Text, Dict, Any, Optional

from . import __version__
from .exceptions import XBotBaseException
from .constants import DEFAULT_MODEL_PATH
from .util.download import download_from_url
from .pipeline import component_config_from_pipeline
from .util.io import read_json_file, write_json_to_file


class InvalidModelError(XBotBaseException):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text) -> None:
        self.message = message
        super(InvalidModelError, self).__init__()

    def __str__(self) -> Text:
        return self.message


class Model(ABC):
    """XBot Model"""

    def __init__(self):
        super(Model, self).__init__()

    @staticmethod
    def load_from_net(url):
        download_from_url(url, DEFAULT_MODEL_PATH)


class ModelMetadata(object):
    """Captures all information about a model to load and prepare it."""

    @staticmethod
    def load(model_dir: Text):
        """Loads the metadata from a models directory.

        Args:
            model_dir: the directory where the model is saved.
        Returns:
            Metadata: A metadata object describing the model
        """
        try:
            metadata_file = os.path.join(model_dir, "metadata.json")
            data = read_json_file(metadata_file)
            return ModelMetadata(data, model_dir)
        except Exception as e:
            abspath = os.path.abspath(os.path.join(model_dir, "metadata.json"))
            raise InvalidModelError(
                f"Failed to load model metadata from '{abspath}'. {e}"
            )

    def __init__(self, metadata: Dict[Text, Any], model_dir: Optional[Text]):

        self.metadata = metadata
        self.model_dir = model_dir

    def get(self, property_name: Text, default: Any = None) -> Any:
        return self.metadata.get(property_name, default)

    @property
    def component_classes(self):
        if self.get("pipeline"):
            return [c.get("class") for c in self.get("pipeline", [])]
        else:
            return []

    @property
    def number_of_components(self):
        return len(self.get("pipeline", []))

    def for_component(self, index: int, defaults: Any = None) -> Dict[Text, Any]:
        return component_config_from_pipeline(index, self.get("pipeline", []), defaults)

    @property
    def language(self) -> Optional[Text]:
        """Language of the underlying model"""

        return self.get("language")

    def persist(self, model_dir: Text):
        """Persists the metadata of a model to a given directory."""

        metadata = self.metadata.copy()

        metadata.update(
            {
                "trained_at": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                "xbot_version": __version__,
            }
        )

        filename = os.path.join(model_dir, "metadata.json")
        write_json_to_file(filename, metadata, indent=4)
