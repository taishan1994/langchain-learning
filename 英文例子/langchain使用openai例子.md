```python
import openai
openai_api_key = ""

# 导入ChatOpenAI，这是LangChain对ChatGPT API访问的抽象
from langchain.chat_models import ChatOpenAI
# 要控制 LLM 生成文本的随机性和创造性，请使用 temperature = 0.0
chat = ChatOpenAI(model_name="gpt-3.5-turbo",
          openai_api_key=openai_api_key,
          temperature=0.0)

# 模板字符串，用于指定目标语言，拥有两个输入变量，"style"和"text"
template_string = """请翻译文本，将其翻译为{language}。文本: ```{text}```"""
# 构建一个ChatPromptTemplate实例，用于模板的重用
from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template_string)

# 将风格和文本作为输入变量传入提示模板
language = "中文"
text = "I don't konw who you are."
customer_messages = prompt_template.format_messages(
                    language="中文",
                    text=text)

customer_response = chat(customer_messages)
print(customer_response.content)

"""
我不知道你是谁。
"""
```

