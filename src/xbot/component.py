import logging
from typing import Any, Dict, Hashable, List, Optional, Set, Text, Tuple, Type, Iterable

from .config import PipelineConfig
from .exceptions import XBotComponentException
from .registry import get_component_class
from .util.config import override_defaults
from .data import TrainingData
from .utterance import Utterance


logger = logging.getLogger(__name__)


class UnsupportedLanguageError(XBotComponentException):
    """Raised when a component is created but the language is not supported.

    Attributes:
        component -- component name
        language -- language that component doesn't support
    """

    def __init__(self, component: Text, language: Text) -> None:
        self.component = component
        self.language = language

        super().__init__(component, language)

    def __str__(self) -> Text:
        return (
            f"component '{self.component}' does not support language '{self.language}'."
        )


class BaseComponent(object):
    """A component is a message processing unit in a pipeline.

    Components are collected sequentially in a pipeline. Each component
    is called one after another. This holds for
    initialization, training, persisting and loading the components.
    If a component comes first in a pipeline, its
    methods will be called first.

    E.g. to process an incoming message, the ``process`` method of
    each component will be called. During the processing
    (as well as the training, persisting and initialization)
    components can pass information to other components.
    The information is passed to other components by providing
    attributes to the so called pipeline context. The
    pipeline context contains all the information of the previous
    components a component can use to do its own
    processing. For example, a featurizer component can provide
    features that are used by another component down
    the pipeline to do intent classification.
    """

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None. if both `support_language_list` and
    # `not_supported_language_list` are None, it means it can handle
    # all languages. Also, only one of `support_language_list` and
    # `not_supported_language_list` can be set to not None.
    # This is an important feature for backwards compatibility of components.
    supported_language_list = None

    # Defines what language(s) this component can NOT handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None. if both `support_language_list` and
    # `not_supported_language_list` are None, it means it can handle
    # all languages. Also, only one of `support_language_list` and
    # `not_supported_language_list` can be set to not None.
    # This is an important feature for backwards compatibility of components.
    not_supported_language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:

        if not component_config:
            component_config = {}

        # makes sure the name of the configuration is part of the config
        # this is important for e.g. persistence
        component_config["name"] = self.name

        self.component_config = override_defaults(self.defaults, component_config)

        self.partial_processing_pipeline = None
        self.partial_processing_context = None

        # Component class name is used when integrating it in a
        # pipeline. E.g. ``[ComponentA, ComponentB]``
        # will be a proper pipeline definition where ``ComponentA``
        # is the name of the first component of the pipeline.

    @property
    def name(self) -> Text:
        """Access the class's property name from an instance."""

        return type(self).name

    @classmethod
    def from_config(
        cls, component_config: Dict[Text, Any], config: "PipelineConfig"
    ) -> Optional["BaseComponent"]:
        """Resolves a component and calls it's create method.

        Inits it based on a previously persisted model.
        """

        # try to get class name first, else create by name
        component_name = component_config.get("class", component_config["name"])
        component_class = get_component_class(component_name)
        return component_class.create(component_config, config)

    # Which components are required by this component.
    # Listed components should appear before the component itself in the pipeline.
    @classmethod
    def required_components(cls) -> List[Type["BaseComponent"]]:
        """Specify which components need to be present in the pipeline.

        Returns:
            The list of class names of required components.
        """

        return []

    @classmethod
    def required_packages(cls) -> List[Text]:
        """Specify which python packages need to be installed.

        E.g. ``["hanlp"]``. More specifically, these should be
        importable python package names e.g. `sklearn` and not package
        names in the dependencies sense e.g. `scikit-learn`

        This list of requirements allows us to fail early during training
        if a required package is not installed.

        Returns:
            The list of required package names.
        """

        return []

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["BaseComponent"] = None,
        **kwargs: Any,
    ) -> "BaseComponent":
        """Load this component from file.

        After a component has been trained, it will be persisted by
        calling `persist`. When the pipeline gets loaded again,
        this component needs to be able to restore itself.
        Components can rely on any context attributes that are
        created by :meth:`components.Component.create`
        calls to components previous to this one.

        Args:
            meta: Any configuration parameter related to the model.
            model_dir: The directory to load the component from.
            model_metadata: The model's :class:`rasa.nlu.model.Metadata`.
            cached_component: The cached component.

        Returns:
            the loaded component
        """

        if cached_component:
            return cached_component

        return cls(meta)

    @classmethod
    def create(
        cls, component_config: Dict[Text, Any], config: PipelineConfig
    ) -> "BaseComponent":
        """Creates this component (e.g. before a training is started).

        Method can access all configuration parameters.

        Args:
            component_config: The components configuration parameters.
            config: The model configuration parameters.

        Returns:
            The created component.
        """

        # Check language supporting
        language = config.language
        if not cls.can_handle_language(language):
            # check failed
            raise UnsupportedLanguageError(cls.name, language)

        return cls(component_config)

    def provide_context(self) -> Optional[Dict[Text, Any]]:
        """Initialize this component for a new pipeline.

        This function will be called before the training
        is started and before the first message is processed using
        the interpreter. The component gets the opportunity to
        add information to the context that is passed through
        the pipeline during training and message parsing. Most
        components do not need to implement this method.
        It's mostly used to initialize framework environments
        like MITIE and spacy
        (e.g. loading word vectors for the pipeline).

        Returns:
            The updated component configuration.
        """

        pass

    def train(
        self,
        training_data: TrainingData,
        config: Optional[PipelineConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.train`
        of components previous to this one.

        Args:
            training_data:
                The :class:`rasa.shared.nlu.training_data.training_data.TrainingData`.
            config: The model configuration parameters.

        """

        pass

    def process(self, message: Utterance, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.

        Args:
            message: The :class:`rasa.shared.nlu.training_data.message.Message` to process.

        """

        pass

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading.

        Args:
            file_name: The file name of the model.
            model_dir: The directory to store the model to.

        Returns:
            An optional dictionary with any information about the stored model.
        """

        pass

    @classmethod
    def cache_key(
        cls, component_meta: Dict[Text, Any], model_metadata: "Metadata"
    ) -> Optional[Text]:
        """This key is used to cache components.

        If a component is unique to a model it should return None.
        Otherwise, an instantiation of the
        component will be reused for all models where the
        metadata creates the same key.

        Args:
            component_meta: The component configuration.
            model_metadata: The component's :class:`rasa.nlu.model.Metadata`.

        Returns:
            A unique caching key.
        """

        return None

    def __getstate__(self) -> Any:
        d = self.__dict__.copy()
        # these properties should not be pickled
        if "partial_processing_context" in d:
            del d["partial_processing_context"]
        if "partial_processing_pipeline" in d:
            del d["partial_processing_pipeline"]
        return d

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__

    def prepare_partial_processing(
        self, pipeline: List["BaseComponent"], context: Dict[Text, Any]
    ) -> None:
        """Sets the pipeline and context used for partial processing.

        The pipeline should be a list of components that are
        previous to this one in the pipeline and
        have already finished their training (and can therefore
        be safely used to process messages).

        Args:
            pipeline: The list of components.
            context: The context of processing.
        """

        self.partial_processing_pipeline = pipeline
        self.partial_processing_context = context

    def partially_process(self, message: Utterance) -> Utterance:
        """Allows the component to process messages during
        training (e.g. external training data).

        The passed message will be processed by all components
        previous to this one in the pipeline.

        Args:
            message: The :class:`rasa.shared.nlu.training_data.message.Message` to
            process.

        Returns:
            The processed :class:`rasa.shared.nlu.training_data.message.Message`.

        """

        if self.partial_processing_context is not None:
            for component in self.partial_processing_pipeline:
                component.process(message, **self.partial_processing_context)
        else:
            logger.info("Failed to run partial processing due to missing pipeline.")
        return message

    @classmethod
    def can_handle_language(cls, language: Hashable) -> bool:
        """Check if component supports a specific language.

        This method can be overwritten when needed. (e.g. dynamically
        determine which language is supported.)

        Args:
            language: The language to check.

        Returns:
            `True` if component can handle specific language, `False` otherwise.
        """

        # If both `supported_language_list` and `not_supported_language_list` are set to `None`,
        # it means: support all languages
        if language is None or (
            cls.supported_language_list is None
            and cls.not_supported_language_list is None
        ):
            return True

        # check language supporting settings
        if cls.supported_language_list and cls.not_supported_language_list:
            # When user set both language supporting settings to not None, it will lead to ambiguity.
            raise XBotComponentException(
                "Only one of `supported_language_list` and `not_supported_language_list` can be set to not None"
            )

        # convert to `list` for membership test
        supported_language_list = (
            cls.supported_language_list
            if cls.supported_language_list is not None
            else []
        )
        not_supported_language_list = (
            cls.not_supported_language_list
            if cls.not_supported_language_list is not None
            else []
        )

        # check if user provided a valid setting
        if not supported_language_list and not not_supported_language_list:
            # One of language settings must be valid (not None and not a empty list),
            # There are three combinations of settings are not valid: (None, []), ([], None) and ([], [])
            raise XBotComponentException(
                "Empty lists for both "
                "`supported_language_list` and `not_supported language_list` "
                "is not a valid setting. If you meant to allow all languages "
                "for the component use `None` for both of them."
            )

        if supported_language_list:
            return language in supported_language_list
        else:
            return language not in not_supported_language_list
