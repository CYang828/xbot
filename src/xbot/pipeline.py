from typing import List, Dict, Optional, Text, Any

from .util.log import raise_warning
from .util.config import override_defaults


class Pipeline(object):
    """对话系统Pipeline"""

    def __init__(self):
        pass


def component_config_from_pipeline(
    index: int,
    pipeline: List[Dict[Text, Any]],
    defaults: Optional[Dict[Text, Any]] = None,
) -> Dict[Text, Any]:
    """Get config of the component with the given index in the pipeline.

    Args:
        index: index the component in the pipeline
        pipeline: a list of component configs in the NLU pipeline
        defaults: default config of the component

    Returns:
        config of the component
    """
    try:
        c = pipeline[index]
        return override_defaults(defaults, c)
    except IndexError:
        raise_warning(
            f"Tried to get configuration value for component "
            f"number {index} which is not part of your pipeline. "
            f"Returning `defaults`."
        )
        return override_defaults(defaults, {})
