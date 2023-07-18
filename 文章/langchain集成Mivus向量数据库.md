```python
import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
from langchain.llms import OpenAI
davinci = OpenAI(model_name="text-davinci-003")
from milvus import default_server
default_server.start()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQA

loader = UnstructuredURLLoader(urls=['https://zilliz.com/doc/about_zilliz_cloud'])
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vector_db = Milvus.from_documents(
   docs,
   embeddings,
   connection_args={"host": "127.0.0.1", "port": default_server.listen_port},
)
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vector_db.as_retriever())
query = "What is Zilliz Cloud?"
qa.run(query)
default_server.stop()
```

现在可以开始学习如何查询文档了。这次从 LangChain 导入了很多内容，需要 OpenAI Embeddings、文本字符拆分器、Milvus 向量数据库、加载器和问答检索链。

- 首先，设置一个加载器并加载 urls 链接中的内容。本例中，将加载 Zilliz Cloud 介绍的文档，即加载链接 'https://zilliz.com/doc/about_zilliz_cloud'。

- 其次，将文档拆分并将其存储为 LangChain 中的一组文档。
- 接着，设置 Milvus 向量数据库。在本例中，我们为刚才通过 `UnstructuredURLLoader`和 `CharacterTextSplitter` 获取的文档数据创建了一个 Milvus 集合（collection）。同时，还使用了 OpenAI Embeddings 将文本转化为 embedding 向量。
- 准备好向量数据库后，可以使用 `RetrievalQA` 通过向量数据库查询文档。使用 `stuff` 类型的链，并选择 OpenAI 作为 LLM，Milvus 向量数据库作为检索器。

接下来，大家就可以查询啦！通过 run 运行查询语句。当然，最后别忘了关闭向量数据库。

原文链接：https://mp.weixin.qq.com/s?__biz=MzUzMDI5OTA5NQ==&mid=2247497707&idx=1&sn=5a8f92cc48ed8f0dae9c1c5739ecc22a&chksm=fa515653cd26df4578d2b5a07e36324698a92ddc86e546cd64ef3294ffec3afd98d8e3d12c23&scene=21#wechat_redirect