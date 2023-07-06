# 总体结构

from langchain.chat_models import ChatOpenAI

ChatOpenAI来源于chat_models，我们去github找到chat_models目录。找到openai.py。里面`class ChatOpenAI(BaseChatModel):`，BaseChatModel来源于chat_models下的base.py，`class BaseChatModel(BaseLanguageModel, ABC)`。BaseLanguageModel来源于langchain目录下的base_language.py。源头已经找到了，我们先去看看base_language.py里面。

# BaseLanguageModel

class BaseLanguageModel(Serializable, ABC):

`from abc import ABC, abstractmethod`

`from langchain.load.serializable import Serializable`

- ABC：子类必须实现某些方法
- Serializable：用于序列化对象

看看BaseLanguageModel具体有哪些抽象方法：

```python
 @abstractmethod
def generate_prompt(
    self,
    prompts: List[PromptValue],
    stop: Optional[List[str]] = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> LLMResult:
    """Take in a list of prompt values and return an LLMResult."""

@abstractmethod
async def agenerate_prompt(
    self,
    prompts: List[PromptValue],
    stop: Optional[List[str]] = None,
    callbacks: Callbacks = None,
    **kwargs: Any,
) -> LLMResult:
    """Take in a list of prompt values and return an LLMResult."""

@abstractmethod
def predict(
    self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
) -> str:
    """Predict text from text."""

@abstractmethod
def predict_messages(
    self,
    messages: List[BaseMessage],
    *,
    stop: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> BaseMessage:
    """Predict message from messages."""

@abstractmethod
async def apredict(
    self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
) -> str:
    """Predict text from text."""

@abstractmethod
async def apredict_messages(
    self,
    messages: List[BaseMessage],
    *,
    stop: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> BaseMessage:
    """Predict message from messages."""
```

在继承BaseLanguageModel必须实现以上方法。我们大可以先不看`async`修饰的这些方法。

# BaseChatModel

我们看看继承BaseLanguageModel的BaseChatModel里面。其确实是实现了上述的方法：

```python
 def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)

```

将prompts转换为message然后再调用 self.generate，看下self.generate是什么：

```python
def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        params = self._get_invocation_params(stop=stop)
        options = {"stop": stop}

        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
        )
        run_managers = callback_manager.on_chat_model_start(
            dumpd(self), messages, invocation_params=params, options=options
        )
        results = []
        for i, m in enumerate(messages):
            try:
                results.append(
                    self._generate_with_cache(
                        m,
                        stop=stop,
                        run_manager=run_managers[i] if run_managers else None,
                        **kwargs,
                    )
                )
            except (KeyboardInterrupt, Exception) as e:
                if run_managers:
                    run_managers[i].on_llm_error(e)
                raise e
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        if run_managers:
            run_infos = []
            for manager, flattened_output in zip(run_managers, flattened_outputs):
                manager.on_llm_end(flattened_output)
                run_infos.append(RunInfo(run_id=manager.run_id))
            output.run = run_infos
        return output
```

里面我们主要关注self._generate_with_cache(，看看它是什么：

```python
def _generate_with_cache(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        new_arg_supported = inspect.signature(self._generate).parameters.get(
            "run_manager"
        )
        disregard_cache = self.cache is not None and not self.cache
        if langchain.llm_cache is None or disregard_cache:
            # This happens when langchain.cache is None, but self.cache is True
            if self.cache is not None and self.cache:
                raise ValueError(
                    "Asked to cache, but no cache found at `langchain.cache`."
                )
            if new_arg_supported:
                return self._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                return self._generate(messages, stop=stop, **kwargs)
        else:
            llm_string = self._get_llm_string(stop=stop, **kwargs)
            prompt = dumps(messages)
            cache_val = langchain.llm_cache.lookup(prompt, llm_string)
            if isinstance(cache_val, list):
                return ChatResult(generations=cache_val)
            else:
                if new_arg_supported:
                    result = self._generate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                else:
                    result = self._generate(messages, stop=stop, **kwargs)
                langchain.llm_cache.update(prompt, llm_string, result.generations)
                return result
```

主要关注self._generate，再去看看：

```python
 @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""

```

这里没有实现，提供给继承BaseLanguageModel的类来实现。也就是说，具体生成的方式是接下来需要定义的。

接下来看看`predict`抽象类：

```python
def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        result = self([HumanMessage(content=text)], stop=_stop, **kwargs)
        return result.content
```

关注这一行：`result = self([HumanMessage(content=text)], stop=_stop, **kwargs)`，实际上调用的是`_call`方法，看看`_call`方法：

```python
 def __call__(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseMessage:
        generation = self.generate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        ).generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        else:
            raise ValueError("Unexpected generation type")
```

里面还是使用`self.generate`，这就又回到了上面所说的一系列调用。

最后看看`predict_messages`：

```python
def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        return self(messages, stop=_stop, **kwargs)
```

同理，调用`_call`。

这里我们发现`_call`是我们的核心，另外生成的主要逻辑位于`self._generate`，需要自行实现。对于其中的一些像回调、消息之类的这些，需要自行去查阅文档理解。

接着看看BaseLanguageModel里面有哪些抽象类需要自类实现：

```python
@abstractmethod
def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
@property
@abstractmethod
def _llm_type(self) -> str:
    """Return type of chat model."""
```

_llm_type是标识该模型的名称。

# ChatOpenAI

最终我们看看openai.py里面的逻辑。`class ChatOpenAI(BaseChatModel):`

ChatOpenAI继承了我们上述的BaseChatModel。

先看看其实现的两个抽象方法：

```python
def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        if self.streaming:
            inner_completion = ""
            role = "assistant"
            params["stream"] = True
            function_call: Optional[dict] = None
            for stream_resp in self.completion_with_retry(
                messages=message_dicts, **params
            ):
                role = stream_resp["choices"][0]["delta"].get("role", role)
                token = stream_resp["choices"][0]["delta"].get("content") or ""
                inner_completion += token
                _function_call = stream_resp["choices"][0]["delta"].get("function_call")
                if _function_call:
                    if function_call is None:
                        function_call = _function_call
                    else:
                        function_call["arguments"] += _function_call["arguments"]
                if run_manager:
                    run_manager.on_llm_new_token(token)
            message = _convert_dict_to_message(
                {
                    "content": inner_completion,
                    "role": role,
                    "function_call": function_call,
                }
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
        response = self.completion_with_retry(messages=message_dicts, **params)
        return self._create_chat_result(response)
    
@property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openai-chat"
```

整体逻辑还是比较明了的，看一下其中的辅助函数。

## self._create_message_dicts

```python
def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params
```

## _convert_message_to_dict

将message转换为字典，主要是解析role（角色）是什么。

```python
def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict
```

## self._invocation_params

传入调用模型时的一些参数。

```python
 @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        openai_creds: Dict[str, Any] = {
            "api_key": self.openai_api_key,
            "api_base": self.openai_api_base,
            "organization": self.openai_organization,
            "model": self.model_name,
        }
        if self.openai_proxy:
            import openai

            openai.proxy = {"http": self.openai_proxy, "https": self.openai_proxy}  # type: ignore[assignment]  # noqa: E501
        return {**openai_creds, **self._default_params}
```

## self.acompletion_with_retry

这里才是真正的调用openai的接口。实际是`llm.client.acreate(**kwargs)`。

```python
def _create_retry_decorator(llm: ChatOpenAI) -> Callable[[Any], Any]:
    import openai

    min_seconds = 1
    max_seconds = 60
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


async def acompletion_with_retry(llm: ChatOpenAI, **kwargs: Any) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        # Use OpenAI's async api https://github.com/openai/openai-python#async-api
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)
```

## self._create_chat_result

解析接口返回的结果，并重新整理为langchain的结果。

```python
 def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(message=message)
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)
```

## _convert_dict_to_message

将字典转换为message。

```python
def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict["content"] or ""  # OpenAI returns None for tool invocations
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)
```

## ChatGeneration

ChatGeneration位于schema目录下的output.py。

```python
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
```

## ChatResult

ChatResult位于schema目录下的output.py。

```python
class ChatResult(BaseModel):
    """Class that contains all results for a single chat model call."""

    generations: List[ChatGeneration]
    """List of the chat generations. This is a List because an input can have multiple 
        candidate generations.
    """
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""
```

至于BaseModel和root_validator都来自`from pydantic import BaseModel, root_validator`。

## pydantic

简单说面一下pydantic：

- pydantic 库是 python 中用于数据接口定义检查与设置管理的库。

- pydantic 在运行时强制执行类型提示，并在数据无效时提供友好的错误。

对于BaseModel，我们看一个例子：

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name = 'Jane Doe'
```

上面的例子，定义了一个User模型，继承自BaseModel，有2个字段，`id`是一个整数并且是必需的，`name`是一个带有默认值的字符串并且不是必需的。

实例化使用：

```python
user = User(id='123')
```

实例化将执行所有解析和验证，如果有错误则会触发 ValidationError 报错。

对于root_validator：是指在类级别上定义的验证函数, 它会在类的所有实例上运行。values包含了模型中的所有参数。需要注意的是，`root_validator`方法的返回值必须是一个字典，其中包含所有验证后的字段值。如果返回的字典中不包含某个字段，则该字段将被设置为默认值或None。

# SimpleChatModel

base.py里面还提供了一个简单的聊天模型定义：

```python
class SimpleChatModel(BaseChatModel):
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Simpler interface."""
```

这里已经实现了`_generate`，我们可以继承SimpleChatModel，再实现`_call`抽象方法即可。

# 总结

到这里，你已经了解了在langchain中的基于openai的相关原理。里面除了各函数之间的调用外，还有一些message和callbacks的使用，这些我们需要通过查阅其官方文档进行进一步的了解。
