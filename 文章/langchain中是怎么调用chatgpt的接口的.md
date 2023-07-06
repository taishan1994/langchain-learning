对于ChatOpenAI，它位于langchain.chat_models下的openai.py里面，我们来看下它是怎么调用的chatgpt的接口的。

ChatOpenAI有一个client属性，在validate_environment进行了赋值`values["client"] = openai.ChatCompletion`，来看下openai.ChatCompletion，其位于langchain目录下的chat_completion.py里面。

```python
import time

from openai import util
from openai.api_resources.abstract.engine_api_resource import EngineAPIResource
from openai.error import TryAgain


class ChatCompletion(EngineAPIResource):
    engine_required = False
    OBJECT_NAME = "chat.completions"

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Creates a new chat completion for the provided messages and parameters.

        See https://platform.openai.com/docs/api-reference/chat-completions/create
        for a list of valid parameters.
        """
        start = time.time()
        timeout = kwargs.pop("timeout", None)

        while True:
            try:
                return super().create(*args, **kwargs)
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)

    @classmethod
    async def acreate(cls, *args, **kwargs):
        """
        Creates a new chat completion for the provided messages and parameters.

        See https://platform.openai.com/docs/api-reference/chat-completions/create
        for a list of valid parameters.
        """
        start = time.time()
        timeout = kwargs.pop("timeout", None)

        while True:
            try:
                return await super().acreate(*args, **kwargs)
            except TryAgain as e:
                if timeout is not None and time.time() > start + timeout:
                    raise

                util.log_info("Waiting for model to warm up", error=e)
```

然后这里使用：

```python
def completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.client.create(**kwargs)

        return _completion_with_retry(**kwargs)
```

在self._generate中进行调用：

```python
response = await acompletion_with_retry(
                self, messages=message_dicts, **params
            )
```

最后，我们去github找到opeai-python库看看它是怎么使用的：

```python
import openai
openai.api_key = "sk-..."

# list models
models = openai.Model.list()

# print the first model's id
print(models.data[0].id)

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world"}])

# print the chat completion
print(chat_completion.choices[0].message.content)
```

基本上就是这么使用的。

我们再尝试下看看结果：

```python
import openai
openai.api_key = "你的api key"

# list models
models = openai.Model.list()

# print the first model's id
total = len(models.data)
for i in range(total):
  print(models.data[i].id)

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "你好，请问你是谁？"}])

# print the chat completion
print(chat_completion.choices[0].message.content)

"""
whisper-1
babbage
text-davinci-003
davinci
text-davinci-edit-001
babbage-code-search-code
text-similarity-babbage-001
code-davinci-edit-001
text-davinci-001
ada
babbage-code-search-text
babbage-similarity
code-search-babbage-text-001
text-curie-001
code-search-babbage-code-001
gpt-3.5-turbo-0613
text-ada-001
text-similarity-ada-001
curie-instruct-beta
gpt-3.5-turbo-0301
gpt-3.5-turbo
ada-code-search-code
ada-similarity
code-search-ada-text-001
text-search-ada-query-001
davinci-search-document
ada-code-search-text
text-search-ada-doc-001
davinci-instruct-beta
text-similarity-curie-001
code-search-ada-code-001
ada-search-query
text-search-davinci-query-001
curie-search-query
davinci-search-query
babbage-search-document
ada-search-document
text-search-curie-query-001
text-search-babbage-doc-001
curie-search-document
text-search-curie-doc-001
babbage-search-query
text-babbage-001
text-search-davinci-doc-001
text-search-babbage-query-001
curie-similarity
gpt-3.5-turbo-16k-0613
curie
text-embedding-ada-002
gpt-3.5-turbo-16k
text-similarity-davinci-001
text-davinci-002
davinci-similarity
你好！我是OpenAI的语言模型，可以回答你的问题和提供一些帮助。
"""
```

我们再去看看流式的结果：

```python
import openai
openai.api_key = "你的api key"

# list models
models = openai.Model.list()

# print the first model's id
total = len(models.data)
for i in range(total):
  print(models.data[i].id)

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "你好，请问你是谁？"}], stream=True)

# print the chat completion
for stream_resp in chat_completion:
  print(stream_resp)
```

```python
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": ""
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u60a8"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u597d"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\uff0c"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u6211"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u662f"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "Open"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "AI"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u7684"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u667a"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u80fd"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u52a9"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u624b"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u3002"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u6211"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u53ef\u4ee5"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u56de"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u7b54"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u60a8"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u7684"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u95ee\u9898"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u548c"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u63d0"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u4f9b"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u4e00"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u4e9b"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u5e2e"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u52a9"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "\u3002"
      },
      "finish_reason": null
    }
  ]
}
{
  "id": "chatcmpl-7ZDPRxw2YxtLBR37F4MUpww3YgLTX",
  "object": "chat.completion.chunk",
  "created": 1688627381,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "delta": {},
      "finish_reason": "stop"
    }
  ]
}
```

它不是一次性返回结果给我们，而是一部分一部分的。

到这里，你已经基本了解了langchain里面是怎么调用openai的chatgpt模型了。