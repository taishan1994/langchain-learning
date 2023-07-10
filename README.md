# langchain-learning
langchain的学习笔记。依赖：

```python
openai==0.27.8
langchian==0.0.225
```

## **文章**

- langchain组件-数据连接(data connection)
- langchain组件-模型IO(model IO)
- langchain组件-链(chains)
- langchain组件-代理(agents)
- langchain组件-内存(memory)
- langchain组件-回调(callbacks)
- langchain中ChatOpenAI背后做了什么.md
- langchain.load.serializable.py.md
- langchain中的一些schema.md
- langchain中是怎么调用chatgpt的接口的.md
- langchain结构化输出背后的原理,md
- langchain中memory的工作原理.md
- pydantic中config的一些配置.md
- pydantic中的Serializable和root_validator.md
- python中常用的一些魔术方法.md
- python的typing常用的类型.md

目前基于langchain的中文项目有两个：

- https://github.com/yanqiangmiffy/Chinese-LangChain
- https://github.com/imClumsyPanda/langchain-ChatGLM

我们从中可以学到不少。

#### langchain-ChatGLM

- 使用api部署langchain-chatglm的基本原理.md

## **中文例子**

- 定制中文LLM模型
- 定制中文聊天模型

## **英文例子**

- langchain使用openai例子.md（文本翻译）
- openai调用chatgpt例子.md
- langchain解析结果并格式化输出.md
- langchain带有记忆的对话.md
- langchain中使用不同链.md
- langchain基于文档的问答md

## **langchain可能存在一些问题**

虽然langchain给我们提供了一些便利，但是也存在一些问题：

- **无法解决大模型基础技术问题，主要是prompt重用问题**：首先很多大模型应用的问题都是大模型基础技术的缺陷，并不是LangChain能够解决的。其中核心的问题是大模型的开发主要工作是prompt工程。而这一点的重用性很低。但是，这些功能都**需要非常定制的手写prompt**。链中的每一步都需要手写prompt。输入数据必须以非常特定的方式格式化，以生成该功能/链步骤的良好输出。设置DAG编排来运行这些链的部分只占工作的5%，95%的工作实际上只是在提示调整和数据序列化格式上。这些东西都是**不可重用**的。

- **LangChain糟糕的抽象与隐藏的垃圾prompt造成开发的困难**：简单说，就是LangChain的抽象工作不够好，所以很多步骤需要自己构建。而且LangChain内置的很多prompt都很差，不如自己构造，但是它们又隐藏了这些默认prompt。
- **LangChain框架很难debug**：**尽管LangChain很多方法提供打印详细信息的参数，但是实际上它们并没有很多有价值的信息**。例如，如果你想看到实际的prompt或者LLM查询等，都是十分困难的。原因和刚才一样，LangChain大多数时候都是隐藏了自己内部的prompt。所以如果你使用LangChain开发效果不好，你想去调试代码看看哪些prompt有问题，那就很难。
- **LangChain鼓励工具锁定**：LangChain鼓励用户在其平台上进行开发和操作，但是如果用户需要进行一些LangChain文档中没有涵盖的工作流程，即使有自定义代理，也很难进行修改。这就意味着，一旦用户开始使用LangChain，他们可能会发现自己被限制在LangChain的特定工具和功能中，而无法轻易地切换到其他可能更适合他们需求的工具或平台。

以上内容来自：

- [Langchain Is Pointless | Hacker News (ycombinator.com)](https://news.ycombinator.com/item?id=36645575)
- [使用LangChain做大模型开发的一些问题：来自Hacker News的激烈讨论~](https://zhuanlan.zhihu.com/p/642498874)

有时候一些简单的任务，我们完全可以自己去实现相关的流程，这样**每一部分都由我们自己把控**，更易于修改。
