# ConversationBufferMemory

存储完整的对话。

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import openai
openai_api_key = ""

# 导入ChatOpenAI，这是LangChain对ChatGPT API访问的抽象
from langchain.chat_models import ChatOpenAI
# 要控制 LLM 生成文本的随机性和创造性，请使用 temperature = 0.0
llm = ChatOpenAI(model_name="gpt-3.5-turbo",
          openai_api_key=openai_api_key,
          temperature=0.0)

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

print(conversation.predict(input="我叫安德鲁"))
print(conversation.predict(input="1+1等于几"))
print(conversation.predict(input="我的名字叫什么"))

"""
> Entering new  chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: 我叫安德鲁
AI:

> Finished chain.
你好，安德鲁！很高兴认识你。我是一个AI助手，我可以回答你的问题或提供帮助。有什么我可以帮你的吗？


> Entering new  chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: 我叫安德鲁
AI: 你好，安德鲁！很高兴认识你。我是一个AI助手，我可以回答你的问题或提供帮助。有什么我可以帮你的吗？
Human: 1+1等于几
AI:

> Finished chain.
1+1等于2。


> Entering new  chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: 我叫安德鲁
AI: 你好，安德鲁！很高兴认识你。我是一个AI助手，我可以回答你的问题或提供帮助。有什么我可以帮你的吗？
Human: 1+1等于几
AI: 1+1等于2。
Human: 我的名字叫什么
AI:

> Finished chain.
你的名字是安德鲁。
"""
```

我们也可以往里面添加新的信息，并打印：

```python
memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})

print(memory.buffer)

"""
> Finished chain.
你的名字是安德鲁。
Human: 我叫安德鲁
AI: 你好，安德鲁！很高兴认识你。我是一个AI助手，我可以回答你的问题或提供帮助。有什么我可以帮你的吗？
Human: 1+1等于几
AI: 1+1等于2。
Human: 我的名字叫什么
AI: 你的名字是安德鲁。
Human: Not much, just hanging
AI: Cool
"""
```

其中input是human信息，ouput是ai信息。

但随着对话的进行，记忆存储的大小会增加，发送Token的成本也会增加，为此，LangChain提供了另外几种策略。

# ConversationBufferWindowMemory

ConversationBufferWindowMemory只保留窗口记忆，也即只保留最后几轮对话。它有一个变量k，表示想记住最后几轮对话。

比如，当k等于1时，表示仅记住最后一轮对话。

例子如下：

```python
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=False
)

conversation.predict(input="Hi, my name is Andrew")
# "Hello Andrew, it's nice to meet you. My name is AI. How can I assist you today?"

conversation.predict(input="What is 1+1?")
# 'The answer to 1+1 is 2.'

conversation.predict(input="What is my name?")
# "I'm sorry, I don't have access to that information. Could you please tell me your name?"





```

这时我们会发现，由于窗口记忆的限制，它会丢失了前面有关名字的交流，从而无法说出我的名字。

这个功能可以防止记忆存储量随着对话的进行而无限增长。当然在实际使用时，k不会设为1，而是会通常设置为一个较大的数字。

# ConversationalTokenBufferMemory

很多LLM定价是基于Token的，Token调用的数量直接反映了LLM调用的成本。

使用ConversationalTokenBufferMemory，可以限制保存在记忆存储的令牌数量。

例子如下：

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI

llm = ChatOpenAI(temperature=0.0)

# 指定LLM和Token限制值
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)

```

在插入一些消息之后，我们可以打印其实际保存的历史消息。

```python
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

# 打印历史消息
memory.load_memory_variables({})
```

我们会发现，当把Token限制值调得比较高时，它几乎可以包含整个对话。

**而如果减少值，它会删掉对话最早的那部分消息，只保留最近对话的消息，并且保证总的消息内容长度不超过Token限制值**。

另外，之所以还要指定一个LLM参数，是因为不同的LLM使用不同的Token计算方式。

这里是告诉它，使用ChatOpenAI LLM使用的计算Token的方法。

# ConversationSummaryBufferMemory

ConversationSummaryBufferMemory试图将消息的显性记忆，保持在我们设定的Token限制值之下，也即

1. 当Token限制值能覆盖文本长度时，会存储整个对话历史。
2. 而当Token限制值小于文本长度时，则会为所有历史消息生成摘要，改在记忆中存储历史消息的摘要。

以情况2为例：

```python
from langchain.memory import ConversationSummaryBufferMemory

# 创建一个长字符串
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
                    
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
conversation.predict(input="What would be a good demo to show?")

"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
Current conversation:
System: The human and AI engage in small talk before discussing the day's schedule. The AI informs the human of a morning meeting with the product team, time to work on the LangChain project, and a lunch meeting with a customer interested in the latest AI developments.
Human: What would be a good demo to show?
AI:
> Finished chain.
"Based on the customer's interest in AI developments, I would suggest showcasing our latest natural language processing capabilities. We could demonstrate how our AI can accurately understand and respond to complex language queries, and even provide personalized recommendations based on the user's preferences. Additionally, we could highlight our AI's ability to learn and adapt over time, making it a valuable tool for businesses looking to improve their customer experience."

"""
```

可以看到，由于超过了设定的Token限制值，它为历史会话的生成了一个摘要，并放在系统消息的提示词中。

# 其它的一些记忆存储策略

- 向量数据存储：存储文本到向量数据库中并检索最相关的文本。
- 实体记忆：在llm中使用，存储特定实体的细节。