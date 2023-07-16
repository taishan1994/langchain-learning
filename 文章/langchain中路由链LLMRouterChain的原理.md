## 基本例子

先看一个基本的例子：

```python
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

print(chain.run("What is black body radiation?"))

    
    
    > Entering new MultiPromptChain chain...
    physics: {'input': 'What is black body radiation?'}
    > Finished chain.
    
    
    Black body radiation is the term used to describe the electromagnetic radiation emitted by a “black body”—an object that absorbs all radiation incident upon it. A black body is an idealized physical body that absorbs all incident electromagnetic radiation, regardless of frequency or angle of incidence. It does not reflect, emit or transmit energy. This type of radiation is the result of the thermal motion of the body's atoms and molecules, and it is emitted at all wavelengths. The spectrum of radiation emitted is described by Planck's law and is known as the black body spectrum.


print(
    chain.run(
        "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3"
    )
)


    
    
    > Entering new MultiPromptChain chain...
    math: {'input': 'What is the first prime number greater than 40 such that one plus the prime number is divisible by 3'}
    > Finished chain.
    ?
    
    The answer is 43. One plus 43 is 44 which is divisible by 3.


print(chain.run("What is the name of the type of cloud that rins"))

    
    
    > Entering new MultiPromptChain chain...
    None: {'input': 'What is the name of the type of cloud that rains?'}
    > Finished chain.
     The type of cloud that rains is called a cumulonimbus cloud. It is a tall and dense cloud that is often accompanied by thunder and lightning.

```

## LLMRouterChain

首先看：from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

class LLMRouterChain(RouterChain)，而from langchain.chains.router.base import RouterChain

class RouterChain(Chain, ABC):，而from langchain.chains.base import Chain

Chain是一个抽象类，有三个抽象方法：

```python
 @property
@abstractmethod
def input_keys(self) -> List[str]: # 链的输入
    """Return the keys expected to be in the chain input."""

@property
@abstractmethod
def output_keys(self) -> List[str]: # 链的输出
    """Return the keys expected to be in the chain output."""
    
@abstractmethod
def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> Dict[str, Any]:
    """Execute the chain.

    This is a private method that is not user-facing. It is only called within
        `Chain.__call__`, which is the user-facing wrapper method that handles
        callbacks configuration and some input/output processing.

    Args:
        inputs: A dict of named inputs to the chain. Assumed to contain all inputs
            specified in `Chain.input_keys`, including any inputs added by memory.
        run_manager: The callbacks manager that contains the callback handlers for
            this run of the chain.

    Returns:
        A dict of named outputs. Should contain all outputs specified in
            `Chain.output_keys`.
    """
```

RouterChain中：

```python
class RouterChain(Chain, ABC):
    """Chain that outputs the name of a destination chain and the inputs to it."""

    @property
    def output_keys(self) -> List[str]:
        return ["destination", "next_inputs"]

    def route(self, inputs: Dict[str, Any], callbacks: Callbacks = None) -> Route:
        result = self(inputs, callbacks=callbacks) # 这里实际上调用的是__call__方法
        return Route(result["destination"], result["next_inputs"])

    async def aroute(
        self, inputs: Dict[str, Any], callbacks: Callbacks = None
    ) -> Route:
        result = await self.acall(inputs, callbacks=callbacks)
        return Route(result["destination"], result["next_inputs"])
```

注意这个Route是什么，

```python
class Route(NamedTuple):
    destination: Optional[str]
    next_inputs: Dict[str, Any]
```

就是一个命名元组，只用于存储信息。

回到LLMRouterChain。

其有一个属性llm_chain: LLMChain，from langchain.chains import LLMChain，之前已经讲解过LLMChain。

实现了抽象方法：

```python
@property
def input_keys(self) -> List[str]:
    """Will be whatever keys the LLM chain prompt expects.

    :meta private:
    """
    return self.llm_chain.input_keys

def _call(
    self,
    inputs: Dict[str, Any],
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> Dict[str, Any]:
    _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
    callbacks = _run_manager.get_child()
    output = cast(
        Dict[str, Any],
        self.llm_chain.predict_and_parse(callbacks=callbacks, **inputs),
    )
    return output

# 有一个类方法
@classmethod
def from_llm(
    cls, llm: BaseLanguageModel, prompt: BasePromptTemplate, **kwargs: Any
) -> LLMRouterChain:
    """Convenience constructor."""
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return cls(llm_chain=llm_chain, **kwargs)
```

用于初始化一个LLMChain，然后设置llm_chain属性。

## MultiPromptChain

MultiPromptChain位于muli_prompt.py中，class MultiPromptChain(MultiRouteChain):，from langchain.chains.router.base import MultiRouteChain

class MultiRouteChain(Chain):，其也继承了Chain，并有以下属性：

```python
"""Use a single chain to route an input to one of multiple candidate chains."""
router_chain: RouterChain    
"""Chain that routes inputs to destination chains."""
destination_chains: Mapping[str, Chain]
"""Chains that return final answer to inputs."""
default_chain: Chain
"""Default chain to use when none of the destination chains are suitable."""
silent_errors: bool = False
"""If True, use default_chain when an invalid destination name is provided. 
Defaults to False."""
```

实现的抽象方法：

```python
@property
def input_keys(self) -> List[str]:
    """Will be whatever keys the router chain prompt expects.

    :meta private:
    """
    return self.router_chain.input_keys

@property
def output_keys(self) -> List[str]:
    """Will always return text key.

    :meta private:
    """
    return []

 def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
    _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
    callbacks = _run_manager.get_child()
    route = self.router_chain.route(inputs, callbacks=callbacks)

    _run_manager.on_text(
        str(route.destination) + ": " + str(route.next_inputs), verbose=self.verbose
    )
    if not route.destination:
        return self.default_chain(route.next_inputs, callbacks=callbacks)
    elif route.destination in self.destination_chains:
        return self.destination_chains[route.destination](
            route.next_inputs, callbacks=callbacks
        )
    elif self.silent_errors:
        return self.default_chain(route.next_inputs, callbacks=callbacks)
    else:
        raise ValueError(
            f"Received invalid destination chain name '{route.destination}'"
```

这里根据router_chain里面的Route里面的值调用不同的chain。

回到MultiRouteChain，

其有以下属性：

```python
"""A multi-route chain that uses an LLM router chain to choose amongst prompts."""
router_chain: RouterChain
"""Chain for deciding a destination chain and the input to it."""
destination_chains: Mapping[str, LLMChain]
"""Map of name to candidate chains that inputs can be routed to."""
default_chain: LLMChain
"""Default chain to use when router doesn't map input to one of the destinations."""
```

分别表示路由链，目标链，默认链。

实现的抽象方法：

```python
@property
def output_keys(self) -> List[str]:
    return ["text"]

# 一个类方法
@classmethod
def from_prompts(
    cls,
    llm: BaseLanguageModel,
    prompt_infos: List[Dict[str, str]],
    default_chain: Optional[LLMChain] = None,
    **kwargs: Any,
) -> MultiPromptChain:
    """Convenience constructor for instantiating from destination prompts."""
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
        destinations=destinations_str
    )
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain
    _default_chain = default_chain or ConversationChain(llm=llm, output_key="text")
    return cls(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=_default_chain,
        **kwargs,
    )
```

根据路由链里面的结果调用其它的一些链。

## MULTI_PROMPT_ROUTER_TEMPLATE

````python
MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT >>
"""
````

就是把一些链的名称和描述填充到destinations中。

## PromptTemplate

class PromptTemplate(StringPromptTemplate):，class StringPromptTemplate(BasePromptTemplate, ABC):，class BasePromptTemplate(Serializable, ABC):

类实例化时：

```python
template=router_template,
input_variables=["input"],
output_parser=RouterOutputParser(),
```

## RouterOutputParser

```python
class RouterOutputParser(BaseOutputParser[Dict[str, str]]):
    """Parser for output of router chain int he multi-prompt chain."""

    default_destination: str = "DEFAULT"
    next_inputs_type: Type = str
    next_inputs_inner_key: str = "input"

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            expected_keys = ["destination", "next_inputs"]
            parsed = parse_and_check_json_markdown(text, expected_keys)
            if not isinstance(parsed["destination"], str):
                raise ValueError("Expected 'destination' to be a string.")
            if not isinstance(parsed["next_inputs"], self.next_inputs_type):
                raise ValueError(
                    f"Expected 'next_inputs' to be {self.next_inputs_type}."
                )
            parsed["next_inputs"] = {self.next_inputs_inner_key: parsed["next_inputs"]}
            if (
                parsed["destination"].strip().lower()
                == self.default_destination.lower()
            ):
                parsed["destination"] = None
            else:
                parsed["destination"] = parsed["destination"].strip()
            return parsed
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )
```

