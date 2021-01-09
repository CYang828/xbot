import logging
import traceback
from typing import Text, Type

from .component import BaseComponent
from .exceptions import XBotComponentException
from .util.common import class_from_module_path


logger = logging.getLogger(__name__)


# Classes of all known components. If a new component should be added,
# its class name should be listed here.
component_classes = []

# Mapping from a components name to its class to allow name based lookup.
registered_components = {c.name: c for c in component_classes}


class ComponentNotFoundException(ModuleNotFoundError, XBotComponentException):
    """Raised if a module referenced by name can not be imported."""


def get_component_class(component_name: Text) -> Type["BaseComponent"]:
    """Resolve component name to a registered components class."""

    if component_name not in registered_components:
        try:
            return class_from_module_path(component_name)

        except (ImportError, AttributeError) as e:
            # when component_name is a path to a class but that path is invalid or
            # when component_name is a class name and not part of old_style_names

            is_path = "." in component_name

            if is_path:
                module_name, _, class_name = component_name.rpartition(".")
                if isinstance(e, ImportError):
                    exception_message = f"Failed to find module '{module_name}'."
                else:
                    # when component_name is a path to a class but the path does
                    # not contain that class
                    exception_message = (
                        f"The class '{class_name}' could not be "
                        f"found in module '{module_name}'."
                    )
            else:
                exception_message = (
                    f"Cannot find class '{component_name}' in global namespace. "
                    f"Please check that there is no typo in the class "
                    f"name and that you have imported the class into the global "
                    f"namespace."
                )

            raise ComponentNotFoundException(
                f"Failed to load the component "
                f"'{component_name}'. "
                f"{exception_message} Either your "
                f"pipeline configuration contains an error "
                f"or the module you are trying to import "
                f"is broken (e.g. the module is trying "
                f"to import a package that is not "
                f"installed). {traceback.format_exc()}"
            )

    return registered_components[component_name]
