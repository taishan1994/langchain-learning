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

```json { "gift": true, "delivery_days": -1, "price_value": ["it is worth $20"] } `

# 结果解析为字典
output_dict = output_parser.parse(response.content)
print(output_dict.get('delivery_days'))``
```

