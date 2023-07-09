先看整体代码：

```python
import openai
openai_api_key = ""

# 导入ChatOpenAI，这是LangChain对ChatGPT API访问的抽象
from langchain.chat_models import ChatOpenAI
# 要控制 LLM 生成文本的随机性和创造性，请使用 temperature = 0.0
chat = ChatOpenAI(model_name="gpt-3.5-turbo",
          openai_api_key=openai_api_key,
          temperature=0.0)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

# 礼物规范
gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
# 送货日期规范
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
# 价格值规范
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

# 将格式规范放到一个列表里
response_schemas = [gift_schema,
                    delivery_days_schema,
                    price_value_schema]
# 构建一个StructuredOutputParser实例
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 获取将发送给LLM的格式指令
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

from langchain.prompts.chat import ChatPromptTemplate

# 提示
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

customer_review = "I bought a wallet as a gift, it is worth $20 and is expected to be delivered on January 3, 2020."

# 构建一个ChatPromptTemplate实例，用于模板的重用
prompt = ChatPromptTemplate.from_template(template=review_template_2)
# 将文本和格式指令作为输入变量传入
messages = prompt.format_messages(text=customer_review,
                  format_instructions=format_instructions)
response = chat(messages)
print(response.content)

```json { "gift": true, "delivery_days": -1, "price_value": ["it is worth $20"] } ```
```

定义了三个schema，然后组成列表。

先看看这两行代码：

```python
# 构建一个StructuredOutputParser实例
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 获取将发送给LLM的格式指令
format_instructions = output_parser.get_format_instructions()
```

StructuredOutputParser位于Langchain的ouput_parsers下，实际上在`__init__.py`里面引入的是`from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser`。

```python
class ResponseSchema(BaseModel):
    name: str
    description: str
    type: str = "string"
class StructuredOutputParser(BaseOutputParser):
    response_schemas: List[ResponseSchema]

    @classmethod
    def from_response_schemas(
        cls, response_schemas: List[ResponseSchema]
    ) -> StructuredOutputParser:
        return cls(response_schemas=response_schemas)

    def get_format_instructions(self, only_json: bool = False) -> str:
        """
        Method to get the format instructions for the output parser.

        example:
        ```python
        from langchain.output_parsers.structured import (
            StructuredOutputParser, ResponseSchema
        )

        response_schemas = [
            ResponseSchema(
                name="foo",
                description="a list of strings",
                type="List[string]"
                ),
            ResponseSchema(
                name="bar",
                description="a string",
                type="string"
                ),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        print(parser.get_format_instructions())

        output:
        # The output should be a markdown code snippet formatted in the following
        # schema, including the leading and trailing "```json" and "```":
        #
        # ```json
        # {
        #     "foo": List[string]  // a list of strings
        #     "bar": string  // a string
        # }

        Args:
            only_json (bool): If True, only the json in the markdown code snippet
                will be returned, without the introducing text. Defaults to False.
        """
        schema_str = "\n".join(
            [_get_sub_string(schema) for schema in self.response_schemas]
        )
        if only_json:
            return STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS.format(format=schema_str)
        else:
            return STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema_str)

    def parse(self, text: str) -> Any:
        expected_keys = [rs.name for rs in self.response_schemas]
        return parse_and_check_json_markdown(text, expected_keys)

    @property
    def _type(self) -> str:
        return "structured"

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
```

上述把用到的都放在一起了。BaseLLMOutputParser与一个抽象方法，返回一个生成器组成的列表。生成器是可能的不同候选输出。BaseOutputParser需要实现两个抽象方法：parse_result、parse。

回到StructuredOutputParser，调用from_response_schemas方法时，初始化其response_schemas属性。

然后调用get_format_instructions。

```python
def _get_sub_string(schema: ResponseSchema) -> str:
    return line_template.format(
        name=schema.name, description=schema.description, type=schema.type
    )
line_template = '\t"{name}": {type}  // {description}'
```

将response_schema中的名称、类型和描述转换为字符串。再来看：

```python
if only_json:
    return STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS.format(format=schema_str)
else:
    return STRUCTURED_FORMAT_INSTRUCTIONS.format(format=schema_str)

STRUCTURED_FORMAT_INSTRUCTIONS = """The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{{
{format}
}}
```"""

STRUCTURED_FORMAT_SIMPLE_INSTRUCTIONS = """
```json
{{
{format}
}}
"""
```

至于ChatPromptTemplate之前已经讲过，这里不作过多讲解，主要是获得prompt。

接下来：

```python
messages = prompt.format_messages(text=customer_review,
                  format_instructions=format_instructions)
```

组合成一个完整的消息后传给chat。

最后：

```python
# 结果解析为字典
output_dict = output_parser.parse(response.content)
print(output_dict.get('delivery_days'))``
```

整体流程就是这样了。

