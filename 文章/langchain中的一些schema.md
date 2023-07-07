langchain中有以下一些schema：

![image-20230706174323824](看看langchain中的一些schema.assets/image-20230706174323824.png)

通过了解这些schema，我们可以知道不同组件的一些属性。大多的时候，我们可以从属性的描述中知道这些schema的用途。

# agent.py

代理相关。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Union


@dataclass
class AgentAction:
    """A full description of an action for an ActionAgent to execute."""

    tool: str
    """The name of the Tool to execute."""
    tool_input: Union[str, dict]
    """The input to pass in to the Tool."""
    log: str
    """Additional information to log about the action."""


class AgentFinish(NamedTuple):
    """The final return value of an ActionAgent."""

    return_values: dict
    """Dictionary of return values."""
    log: str
    """Additional information to log about the return value"""
```

这里面有一个from dataclasses import dataclass，我们看看它是什么：

## dataclass

dataclass 是一个 Python 3.7 引入的装饰器，用于简化定义数据类的过程。通过使用 dataclass 装饰器，我们可以更容易地定义一些简单的数据类，而不需要手动编写大量的代码。

使用 dataclass 装饰器定义一个数据类非常简单，只需要使用 @dataclass 装饰器来修饰类，并在类中定义一个或多个属性即可。

以下是一个使用 dataclass 定义一个简单数据类的示例：

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    is_student: bool = False
```

在该示例中，我们使用 @dataclass 装饰器修饰了一个名为 Person 的类，并定义了三个属性：name、age 和 is_student。其中，name 和 age 属性都是必需的，is_student 属性是可选的，并且默认值为 False。

使用 dataclass 装饰器修饰的类，会自动生成一些特殊方法，例如 __init__()、__repr__()、__eq__() 等，这些方法用于初始化类的实例、打印类的实例、比较类的实例等等。我们可以直接使用这些方法，而不需要手动编写它们。

以下是一个使用 Person 类的示例：

```python
# 创建 Person 对象
person = Person(name="Alice", age=25, is_student=True)

# 打印 Person 对象的属性
print(person.name)
print(person.age)
print(person.is_student)

# 打印 Person 对象的字符串表示形式
print(person)
```

在该示例中，我们创建了一个 Person 对象，并打印了其属性和字符串表示形式。可以看到，使用 dataclass 装饰器定义的 Person 类非常简洁，并且自动生成了一些特殊方法，使得我们可以更方便地使用这个类。

# document

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from pydantic import Field

from langchain.load.serializable import Serializable


class Document(Serializable):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """


class BaseDocumentTransformer(ABC):
    """Abstract base class for document transformation systems.

    A document transformation system takes a sequence of Documents and returns a
    sequence of transformed Documents.

    Example:
        .. code-block:: python

            class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
                embeddings: Embeddings
                similarity_fn: Callable = cosine_similarity
                similarity_threshold: float = 0.95

                class Config:
                    arbitrary_types_allowed = True

                def transform_documents(
                    self, documents: Sequence[Document], **kwargs: Any
                ) -> Sequence[Document]:
                    stateful_documents = get_stateful_documents(documents)
                    embedded_documents = _get_embeddings_from_stateful_docs(
                        self.embeddings, stateful_documents
                    )
                    included_idxs = _filter_similar_embeddings(
                        embedded_documents, self.similarity_fn, self.similarity_threshold
                    )
                    return [stateful_documents[i] for i in sorted(included_idxs)]

                async def atransform_documents(
                    self, documents: Sequence[Document], **kwargs: Any
                ) -> Sequence[Document]:
                    raise NotImplementedError

    """  # noqa: E501

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """

    @abstractmethod
    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of transformed Documents.
        """
```

这里有用到from langchain.load.serializable import Serializable，看看那是什么：

```python
from abc import ABC
from typing import Any, Dict, List, Literal, TypedDict, Union, cast

from pydantic import BaseModel, PrivateAttr


class BaseSerialized(TypedDict):
    """Base class for serialized objects."""

    lc: int
    id: List[str]


class SerializedConstructor(BaseSerialized):
    """Serialized constructor."""

    type: Literal["constructor"]
    kwargs: Dict[str, Any]


class SerializedSecret(BaseSerialized):
    """Serialized secret."""

    type: Literal["secret"]


class SerializedNotImplemented(BaseSerialized):
    """Serialized not implemented."""

    type: Literal["not_implemented"]


class Serializable(BaseModel, ABC):
    """Serializable base class."""

    @property
    def lc_serializable(self) -> bool:
        """
        Return whether or not the class is serializable.
        """
        return False

    @property
    def lc_namespace(self) -> List[str]:
        """
        Return the namespace of the langchain object.
        eg. ["langchain", "llms", "openai"]
        """
        return self.__class__.__module__.split(".")

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """
        Return a map of constructor argument names to secret ids.
        eg. {"openai_api_key": "OPENAI_API_KEY"}
        """
        return dict()

    @property
    def lc_attributes(self) -> Dict:
        """
        Return a list of attribute names that should be included in the
        serialized kwargs. These attributes must be accepted by the
        constructor.
        """
        return {}

    class Config:
        extra = "ignore"

    _lc_kwargs = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lc_kwargs = kwargs

    def to_json(self) -> Union[SerializedConstructor, SerializedNotImplemented]:
        if not self.lc_serializable:
            return self.to_json_not_implemented()

        secrets = dict()
        # Get latest values for kwargs if there is an attribute with same name
        lc_kwargs = {
            k: getattr(self, k, v)
            for k, v in self._lc_kwargs.items()
            if not (self.__exclude_fields__ or {}).get(k, False)  # type: ignore
        }

        # Merge the lc_secrets and lc_attributes from every class in the MRO
        for cls in [None, *self.__class__.mro()]:
            # Once we get to Serializable, we're done
            if cls is Serializable:
                break

            # Get a reference to self bound to each class in the MRO
            this = cast(Serializable, self if cls is None else super(cls, self))

            secrets.update(this.lc_secrets)
            lc_kwargs.update(this.lc_attributes)

        # include all secrets, even if not specified in kwargs
        # as these secrets may be passed as an environment variable instead
        for key in secrets.keys():
            secret_value = getattr(self, key, None) or lc_kwargs.get(key)
            if secret_value is not None:
                lc_kwargs.update({key: secret_value})

        return {
            "lc": 1,
            "type": "constructor",
            "id": [*self.lc_namespace, self.__class__.__name__],
            "kwargs": lc_kwargs
            if not secrets
            else _replace_secrets(lc_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        return to_json_not_implemented(self)


def _replace_secrets(
    root: Dict[Any, Any], secrets_map: Dict[str, str]
) -> Dict[Any, Any]:
    result = root.copy()
    for path, secret_id in secrets_map.items():
        [*parts, last] = path.split(".")
        current = result
        for part in parts:
            if part not in current:
                break
            current[part] = current[part].copy()
            current = current[part]
        if last in current:
            current[last] = {
                "lc": 1,
                "type": "secret",
                "id": [secret_id],
            }
    return result


def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """Serialize a "not implemented" object.

    Args:
        obj: object to serialize

    Returns:
        SerializedNotImplemented
    """
    _id: List[str] = []
    try:
        if hasattr(obj, "__name__"):
            _id = [*obj.__module__.split("."), obj.__name__]
        elif hasattr(obj, "__class__"):
            _id = [*obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        pass
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": _id,
    }
```

稍后我们单独讲讲这个。

# memory

内存相关，用于记录对话中的一些消息。

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain.load.serializable import Serializable
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage


class BaseMemory(Serializable, ABC):
    """Base abstract class for memory in Chains.

    Memory refers to state in Chains. Memory can be used to store information about
        past executions of a Chain and inject that information into the inputs of
        future executions of the Chain. For example, for conversational Chains Memory
        can be used to store conversations and automatically add them to future model
        prompts so that the model has the necessary context to respond coherently to
        the latest input.

    Example:
        .. code-block:: python

            class SimpleMemory(BaseMemory):
                memories: Dict[str, Any] = dict()

                @property
                def memory_variables(self) -> List[str]:
                    return list(self.memories.keys())

                def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                    return self.memories

                def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
                    pass

                def clear(self) -> None:
                    pass
    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """The string keys this memory class will add to chain inputs."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this chain run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""


class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    See `ChatMessageHistory` for default implementation.

    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: List[BaseMessage]
    """A list of Messages stored in-memory."""

    def add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        self.add_message(AIMessage(content=message))

    # TODO: Make this an abstractmethod.
    def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store"""
```

# message

对话中不同角色的一些消息定义。

```python
from __future__ import annotations

from abc import abstractmethod
from typing import List, Sequence

from pydantic import Field

from langchain.load.serializable import Serializable


def get_buffer_string(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Convert sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
        ai_prefix: THe prefix to prepend to contents of AIMessages.

    Returns:
        A single string concatenation of all input messages.

    Example:
        .. code-block:: python

            from langchain.schema import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="Hi, how are you?"),
                AIMessage(content="Good, how are you?"),
            ]
            get_buffer_string(messages)
            # -> "Human: Hi, how are you?\nAI: Good, how are you?"
    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"{role}: {m.content}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)


class BaseMessage(Serializable):
    """The base abstract Message class.

    Messages are the inputs and outputs of ChatModels.
    """

    content: str
    """The string contents of the message."""

    additional_kwargs: dict = Field(default_factory=dict)
    """Any additional information."""

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the Message, used for serialization."""

    @property
    def lc_serializable(self) -> bool:
        """Whether this class is LangChain serializable."""
        return True


class HumanMessage(BaseMessage):
    """A Message from a human."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "human"


class AIMessage(BaseMessage):
    """A Message from an AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ai"


class SystemMessage(BaseMessage):
    """A Message for priming AI behavior, usually passed in as the first of a sequence
    of input messages.
    """

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "system"


class FunctionMessage(BaseMessage):
    """A Message for passing the result of executing a function back to a model."""

    name: str
    """The name of the function that was executed."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "function"


class ChatMessage(BaseMessage):
    """A Message that can be assigned an arbitrary speaker (i.e. role)."""

    role: str
    """The speaker / role of the Message."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "chat"


def _message_to_dict(message: BaseMessage) -> dict:
    return {"type": message.type, "data": message.dict()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> List[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as BaseMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [_message_to_dict(m) for m in messages]


def _message_from_dict(message: dict) -> BaseMessage:
    _type = message["type"]
    if _type == "human":
        return HumanMessage(**message["data"])
    elif _type == "ai":
        return AIMessage(**message["data"])
    elif _type == "system":
        return SystemMessage(**message["data"])
    elif _type == "chat":
        return ChatMessage(**message["data"])
    else:
        raise ValueError(f"Got unexpected type: {_type}")


def messages_from_dict(messages: List[dict]) -> List[BaseMessage]:
    """Convert a sequence of messages from dicts to Message objects.

    Args:
        messages: Sequence of messages (as dicts) to convert.

    Returns:
        List of messages (BaseMessages).
    """
    return [_message_from_dict(m) for m in messages]
```

# output

对输出的一些属性。

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, root_validator

from langchain.load.serializable import Serializable
from langchain.schema.messages import BaseMessage


class Generation(Serializable):
    """A single text generation output."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw response from the provider. May include things like the 
        reason for finishing or token log probabilities.
    """
    # TODO: add log probs as separate attribute

    @property
    def lc_serializable(self) -> bool:
        """Whether this class is LangChain serializable."""
        return True


class ChatGeneration(Generation):
    """A single chat generation output."""

    text: str = ""
    """*SHOULD NOT BE SET DIRECTLY* The text contents of the output message."""
    message: BaseMessage
    """The message output by the chat model."""

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set the text attribute to be the contents of the message."""
        values["text"] = values["message"].content
        return values


class RunInfo(BaseModel):
    """Class that contains metadata for a single execution of a Chain or model."""

    run_id: UUID
    """A unique identifier for the model or chain run."""


class ChatResult(BaseModel):
    """Class that contains all results for a single chat model call."""

    generations: List[ChatGeneration]
    """List of the chat generations. This is a List because an input can have multiple 
        candidate generations.
    """
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class LLMResult(BaseModel):
    """Class that contains all results for a batched LLM call."""

    generations: List[List[Generation]]
    """List of generated outputs. This is a List[List[]] because
    each input could have multiple candidate generations."""
    llm_output: Optional[dict] = None
    """Arbitrary LLM provider-specific output."""
    run: Optional[List[RunInfo]] = None
    """List of metadata info for model call for each input."""

    def flatten(self) -> List[LLMResult]:
        """Flatten generations into a single list.

        Unpack List[List[Generation]] -> List[LLMResult] where each returned LLMResult
            contains only a single Generation. If token usage information is available,
            it is kept only for the LLMResult corresponding to the top-choice
            Generation, to avoid over-counting of token usage downstream.

        Returns:
            List of LLMResults where each returned LLMResult contains a single
                Generation.
        """
        llm_results = []
        for i, gen_list in enumerate(self.generations):
            # Avoid double counting tokens in OpenAICallback
            if i == 0:
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=self.llm_output,
                    )
                )
            else:
                if self.llm_output is not None:
                    llm_output = deepcopy(self.llm_output)
                    llm_output["token_usage"] = dict()
                else:
                    llm_output = None
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=llm_output,
                    )
                )
        return llm_results

    def __eq__(self, other: object) -> bool:
        """Check for LLMResult equality by ignoring any metadata related to runs."""
        if not isinstance(other, LLMResult):
            return NotImplemented
        return (
            self.generations == other.generations
            and self.llm_output == other.llm_output
        )
```

# output_parser

用于解析输出。

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from langchain.load.serializable import Serializable
from langchain.schema.output import Generation
from langchain.schema.prompt import PromptValue

T = TypeVar("T")


class BaseLLMOutputParser(Serializable, ABC, Generic[T]):
    """Abstract base class for parsing the outputs of a model."""

    @abstractmethod
    def parse_result(self, result: List[Generation]) -> T:
        """Parse a list of candidate model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """


class BaseOutputParser(BaseLLMOutputParser, ABC, Generic[T]):
    """Class to parse the output of an LLM call.

    Output parsers help structure language model responses.

    Example:
        .. code-block:: python

            class BooleanOutputParser(BaseOutputParser[bool]):
                true_val: str = "YES"
                false_val: str = "NO"

                def parse(self, text: str) -> bool:
                    cleaned_text = text.strip().upper()
                    if cleaned_text not in (self.true_val.upper(), self.false_val.upper()):
                        raise OutputParserException(
                            f"BooleanOutputParser expected output value to either be "
                            f"{self.true_val} or {self.false_val} (case-insensitive). "
                            f"Received {cleaned_text}."
                        )
                    return cleaned_text == self.true_val.upper()

                    @property
                    def _type(self) -> str:
                            return "boolean_output_parser"
    """  # noqa: E501

    def parse_result(self, result: List[Generation]) -> T:
        """Parse a list of candidate model Generations into a specific format.

        The return value is parsed from only the first Generation in the result, which
            is assumed to be the highest-likelihood Generation.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.

        Returns:
            Structured output.
        """
        return self.parse(result[0].text)

    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse a single string model output into some structure.

        Args:
            text: String output of language model.

        Returns:
            Structured output.
        """

    # TODO: rename 'completion' -> 'text'.
    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
        """Parse the output of an LLM call with the input prompt for context.

        The prompt is largely provided in the event the OutputParser wants
        to retry or fix the output in some way, and needs information from
        the prompt to do so.

        Args:
            completion: String output of language model.
            prompt: Input PromptValue.

        Returns:
            Structured output
        """
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        raise NotImplementedError(
            f"_type property is not implemented in class {self.__class__.__name__}."
            " This is required for serialization."
        )

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of output parser."""
        output_parser_dict = super().dict(**kwargs)
        output_parser_dict["_type"] = self._type
        return output_parser_dict


class NoOpOutputParser(BaseOutputParser[str]):
    """'No operation' OutputParser that returns the text as is."""

    @property
    def lc_serializable(self) -> bool:
        """Whether the class LangChain serializable."""
        return True

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "default"

    def parse(self, text: str) -> str:
        """Returns the input text with no changes."""
        return text


class OutputParserException(ValueError):
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser. OutputParserExceptions will be
    available to catch and handle in ways to fix the parsing error, while other
    errors will be raised.

    Args:
        error: The error that's being re-raised or an error message.
        observation: String explanation of error which can be passed to a
            model to try and remediate the issue.
        llm_output: String model output which is error-ing.
        send_to_llm: Whether to send the observation and llm_output back to an Agent
            after an OutputParserException has been raised. This gives the underlying
            model driving the agent the context that the previous output was improperly
            structured, in the hopes that it will update the output to the correct
            format.
    """

    def __init__(
        self,
        error: Any,
        observation: Optional[str] = None,
        llm_output: Optional[str] = None,
        send_to_llm: bool = False,
    ):
        super(OutputParserException, self).__init__(error)
        if send_to_llm:
            if observation is None or llm_output is None:
                raise ValueError(
                    "Arguments 'observation' & 'llm_output'"
                    " are required if 'send_to_llm' is True"
                )
        self.observation = observation
        self.llm_output = llm_output
        self.send_to_llm = send_to_llm
```

# prompt

输入的prompt的一些属性。

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain.load.serializable import Serializable
from langchain.schema.messages import BaseMessage


class PromptValue(Serializable, ABC):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    @abstractmethod
    def to_string(self) -> str:
        """Return prompt value as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
```

# prompt_template

这里面有一个BasePromptTemplate，这是我们自定义prompt模板所需要的。

```python
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import yaml
from pydantic import Field, root_validator

from langchain.load.serializable import Serializable
from langchain.schema.document import Document
from langchain.schema.output_parser import BaseOutputParser
from langchain.schema.prompt import PromptValue


class BasePromptTemplate(Serializable, ABC):
    """Base class for all prompt templates, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""
    output_parser: Optional[BaseOutputParser] = None
    """How to parse the output of calling an LLM on this formatted prompt."""
    partial_variables: Mapping[str, Union[str, Callable[[], str]]] = Field(
        default_factory=dict
    )

    @property
    def lc_serializable(self) -> bool:
        return True

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create Chat Messages."""

    @root_validator()
    def validate_variable_names(cls, values: Dict) -> Dict:
        """Validate variable names do not include restricted names."""
        if "stop" in values["input_variables"]:
            raise ValueError(
                "Cannot have an input variable named 'stop', as it is used internally,"
                " please rename."
            )
        if "stop" in values["partial_variables"]:
            raise ValueError(
                "Cannot have an partial variable named 'stop', as it is used "
                "internally, please rename."
            )

        overall = set(values["input_variables"]).intersection(
            values["partial_variables"]
        )
        if overall:
            raise ValueError(
                f"Found overlapping input and partial variables: {overall}"
            )
        return values

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        """Return a partial of the prompt template."""
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        # Get partial params:
        partial_kwargs = {
            k: v if isinstance(v, str) else v()
            for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of prompt."""
        prompt_dict = super().dict(**kwargs)
        prompt_dict["_type"] = self._prompt_type
        return prompt_dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the prompt.

        Args:
            file_path: Path to directory to save prompt to.

        Example:
        .. code-block:: python

            prompt.save(file_path="path/prompt.yaml")
        """
        if self.partial_variables:
            raise ValueError("Cannot save prompt with partial variables.")
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")


def format_document(doc: Document, prompt: BasePromptTemplate) -> str:
    """Format a document into a string based on a prompt template.

    First, this pulls information from the document from two sources:

    1. `page_content`:
        This takes the information from the `document.page_content`
        and assigns it to a variable named `page_content`.
    2. metadata:
        This takes information from `document.metadata` and assigns
        it to variables of the same name.

    Those variables are then passed into the `prompt` to produce a formatted string.

    Args:
        doc: Document, the page_content and metadata will be used to create
            the final string.
        prompt: BasePromptTemplate, will be used to format the page_content
            and metadata into the final string.

    Returns:
        string of the document formatted.

    Example:
        .. code-block:: python

            from langchain.schema import Document
            from langchain.prompts import PromptTemplate
            doc = Document(page_content="This is a joke", metadata={"page": "1"})
            prompt = PromptTemplate.from_template("Page {page}: {page_content}")
            format_document(doc, prompt)
            >>> "Page 1: This is a joke"
    """
    base_info = {"page_content": doc.page_content, **doc.metadata}
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        raise ValueError(
            f"Document prompt requires documents to have metadata variables: "
            f"{required_metadata}. Received document with missing metadata: "
            f"{list(missing_metadata)}."
        )
    document_info = {k: base_info[k] for k in prompt.input_variables}
    return prompt.format(**document_info)
```

# retriever

和检索相关，BaseRetriever可能是自定义检索器所需要的，暂时没有看到这方面的使用，可能以后会发现。

```python
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.load.dump import dumpd
from langchain.load.serializable import Serializable
from langchain.schema.document import Document

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
        Callbacks,
    )


class BaseRetriever(Serializable, ABC):
    """Abstract base class for a Document retrieval system.

    A retrieval system is defined as something that can take string queries and return
        the most 'relevant' Documents from some source.

    Example:
        .. code-block:: python

            class TFIDFRetriever(BaseRetriever, BaseModel):
                vectorizer: Any
                docs: List[Document]
                tfidf_array: Any
                k: int = 4

                class Config:
                    arbitrary_types_allowed = True

                def get_relevant_documents(self, query: str) -> List[Document]:
                    from sklearn.metrics.pairwise import cosine_similarity

                    # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                    query_vec = self.vectorizer.transform([query])
                    # Op -- (n_docs,1) -- Cosine Sim with each doc
                    results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
                    return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]

                async def aget_relevant_documents(self, query: str) -> List[Document]:
                    raise NotImplementedError
    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    _new_arg_supported: bool = False
    _expects_other_args: bool = False
    tags: Optional[List[str]] = None
    """Optional list of tags associated with the retriever. Defaults to None
    These tags will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its 
    use case.
    """
    metadata: Optional[Dict[str, Any]] = None
    """Optional metadata associated with the retriever. Defaults to None
    This metadata will be associated with each call to this retriever,
    and passed as arguments to the handlers defined in `callbacks`.
    You can use these to eg identify a specific instance of a retriever with its 
    use case.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Version upgrade for old retrievers that implemented the public
        # methods directly.
        if cls.get_relevant_documents != BaseRetriever.get_relevant_documents:
            warnings.warn(
                "Retrievers must implement abstract `_get_relevant_documents` method"
                " instead of `get_relevant_documents`",
                DeprecationWarning,
            )
            swap = cls.get_relevant_documents
            cls.get_relevant_documents = (  # type: ignore[assignment]
                BaseRetriever.get_relevant_documents
            )
            cls._get_relevant_documents = swap  # type: ignore[assignment]
        if (
            hasattr(cls, "aget_relevant_documents")
            and cls.aget_relevant_documents != BaseRetriever.aget_relevant_documents
        ):
            warnings.warn(
                "Retrievers must implement abstract `_aget_relevant_documents` method"
                " instead of `aget_relevant_documents`",
                DeprecationWarning,
            )
            aswap = cls.aget_relevant_documents
            cls.aget_relevant_documents = (  # type: ignore[assignment]
                BaseRetriever.aget_relevant_documents
            )
            cls._aget_relevant_documents = aswap  # type: ignore[assignment]
        parameters = signature(cls._get_relevant_documents).parameters
        cls._new_arg_supported = parameters.get("run_manager") is not None
        # If a V1 retriever broke the interface and expects additional arguments
        cls._expects_other_args = (
            len(set(parameters.keys()) - {"self", "query", "run_manager"}) > 0
        )

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

    @abstractmethod
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
            tags: Optional list of tags associated with the retriever. Defaults to None
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            metadata: Optional metadata associated with the retriever. Defaults to None
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
        Returns:
            List of relevant documents
        """
        from langchain.callbacks.manager import CallbackManager

        callback_manager = CallbackManager.configure(
            callbacks,
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=tags,
            local_tags=self.tags,
            inheritable_metadata=metadata,
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            query,
            **kwargs,
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents(
                    query, run_manager=run_manager, **_kwargs
                )
            else:
                result = self._get_relevant_documents(query, **_kwargs)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
            tags: Optional list of tags associated with the retriever. Defaults to None
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            metadata: Optional metadata associated with the retriever. Defaults to None
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
        Returns:
            List of relevant documents
        """
        from langchain.callbacks.manager import AsyncCallbackManager

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=tags,
            local_tags=self.tags,
            inheritable_metadata=metadata,
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            dumpd(self),
            query,
            **kwargs,
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = await self._aget_relevant_documents(
                    query, run_manager=run_manager, **_kwargs
                )
            else:
                result = await self._aget_relevant_documents(query, **_kwargs)
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result
```

# 总结

上述的一些schema基本上定义了一些模块里面的属性。另外也定义了模块中的一些基本类，通过继承这些基本类，我们可以自定义相关的模块。同时，对于不同模块的各种属性，我们只需要了解它有这么一个东西，并不需要死记硬背。有需要的时候可以过来翻翻看到底是什么。
