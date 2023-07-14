看了这么多源码以及实例，接下来我们将做一个最小限度的langchain-chatglm：

## 从文本中加载数据

```python
text = """
ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考我们的博客。

为了方便下游开发者针对自己的应用场景定制模型，我们同时实现了基于 P-Tuning v2 的高效参数微调方法 (使用指南) ，INT4 量化级别下最低只需 7GB 显存即可启动微调。

想让 ChatGLM-6B 更符合你的应用场景？欢迎参与 Badcase 反馈计划。

ChatGLM-6B 开源模型旨在与开源社区一起推动大模型技术发展，恳请开发者和大家遵守开源协议，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。目前，本项目团队未基于 ChatGLM-6B 开发任何应用，包括网页端、安卓、苹果 iOS 及 Windows App 等应用。

尽管模型在训练的各个阶段都尽力确保数据的合规性和准确性，但由于 ChatGLM-6B 模型规模较小，且模型受概率随机性因素影响，无法保证输出内容的准确性，且模型易被误导（详见局限性）。本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。

更新信息
[2023/06/25] 发布 ChatGLM2-6B，ChatGLM-6B 的升级版本，在保留了了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 引入了如下新特性：

更强大的性能：基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，评测结果显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。
更长的上下文：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练，允许更多轮次的对话。但当前版本的 ChatGLM2-6B 对单轮超长文档的理解能力有限，我们会在后续迭代升级中着重进行优化。
更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。
更多信息参见 ChatGLM2-6B。

[2023/06/14] 发布 WebGLM，一项被接受于KDD 2023的研究工作，支持利用网络信息生成带有准确引用的长回答。
"""

import logging
from typing import List, Optional, Any

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.helpers import detect_file_encodings

class StrLoader(BaseLoader):
    """Load text files.


    Args:
        file_path: Path to the file to load.

        encoding: File encoding to use. If `None`, the file will be loaded
        with the default system encoding.

        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
    """

    def __init__(
        self,
        text: str,
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ):
        """Initialize with file path."""
        self.text = text
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        # 从文本加载
        metadata = {"source": "chatglm.md"}
        return [Document(page_content=text, metadata=metadata)]

print("文本长度：", len(text))

loader = StrLoader(text)
```

继承BaseLoader，重写load方法，返回一个Document类，其中page_content是我们的文本，metadata是一些额外的元信息，比如文本的路径、文本的描述等。

## 将文本进行分割

```python

from langchain.text_splitter import CharacterTextSplitter

class CharacterStrSplitter(CharacterTextSplitter):
    """Implementation of splitting text that looks at characters."""

    def __init__(self, separator: str = "\n\n", **kwargs: Any) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        splits = text.split(self._separator)
        _separator = "" if self._keep_separator else self._separator
        return self._merge_splits(splits, _separator)

text_splitter = CharacterStrSplitter(        
    separator = "\n\n",
    chunk_size = 512,
    chunk_overlap  = 100,
    length_function = len,
)
docs = loader.load_and_split(text_splitter)
from pprint import pprint 
pprint(docs)

"""
WARNING:langchain.text_splitter:Created a chunk of size 535, which is longer than the specified 512
文本长度： 1442
[Document(page_content='ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考我们的博客。\n\n为了方便下游开发者针对自己的应用场景定制模型，我们同时实现了基于 P-Tuning v2 的高效参数微调方法 (使用指南) ，INT4 量化级别下最低只需 7GB 显存即可启动微调。\n\n想让 ChatGLM-6B 更符合你的应用场景？欢迎参与 Badcase 反馈计划。', metadata={'source': 'chatglm.md'}),
 Document(page_content='想让 ChatGLM-6B 更符合你的应用场景？欢迎参与 Badcase 反馈计划。\n\nChatGLM-6B 开源模型旨在与开源社区一起推动大模型技术发展，恳请开发者和大家遵守开源协议，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。目前，本项目团队未基于 ChatGLM-6B 开发任何应用，包括网页端、安卓、苹果 iOS 及 Windows App 等应用。\n\n尽管模型在训练的各个阶段都尽力确保数据的合规性和准确性，但由于 ChatGLM-6B 模型规模较小，且模型受概率随机性因素影响，无法保证输出内容的准确性，且模型易被误导（详见局限性）。本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。\n\n更新信息\n[2023/06/25] 发布 ChatGLM2-6B，ChatGLM-6B 的升级版本，在保留了了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 引入了如下新特性：', metadata={'source': 'chatglm.md'}),
 Document(page_content='更强大的性能：基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，评测结果显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。\n更长的上下文：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练，允许更多轮次的对话。但当前版本的 ChatGLM2-6B 对单轮超长文档的理解能力有限，我们会在后续迭代升级中着重进行优化。\n更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。\n更多信息参见 ChatGLM2-6B。', metadata={'source': 'chatglm.md'}),
 Document(page_content='[2023/06/14] 发布 WebGLM，一项被接受于KDD 2023的研究工作，支持利用网络信息生成带有准确引用的长回答。', metadata={'source': 'chatglm.md'})]
"""
```

继承CharacterTextSplitter，重写split_text方法，这里我只简单的进行\n\n分割文本，可根据自己的需求定义。

初始化的参数：

- separator：分割符
- chunk_size：文本块的长度
- chunk_overlap：文本块之间重叠的长度
- length_function：计算长度的方法，这里也可以先进行tokenizer化，再计算长度

## 使用模型转换分割后的文本并进行存储

```python
from langchian.
embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese", model_kwargs={'device': ""})
from langchain.vectorstores.faiss import FAISS
vector_store = FAISS.from_documents(docs, embeddings) 

related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)

"""
[(Document(page_content='ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考我们的博客。\n\n为了方便下游开发者针对自己的应用场景定制模型，我们同时实现了基于 P-Tuning v2 的高效参数微调方法 (使用指南) ，INT4 量化级别下最低只需 7GB 显存即可启动微调。\n\n想让 ChatGLM-6B 更符合你的应用场景？欢迎参与 Badcase 反馈计划。', metadata={'source': 'chatglm.md'}),
  488.00125),
 (Document(page_content='更强大的性能：基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，评测结果显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。\n更长的上下文：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练，允许更多轮次的对话。但当前版本的 ChatGLM2-6B 对单轮超长文档的理解能力有限，我们会在后续迭代升级中着重进行优化。\n更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。\n更多信息参见 ChatGLM2-6B。', metadata={'source': 'chatglm.md'}),
  538.00037)]
"""
```

这里可参考已经讲解的文章。

## 构建prompt

```python
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

context = "\n".join([doc[0].page_content for doc in related_docs_with_score])
question = query

prompt = PROMPT_TEMPLATE.replace("{question}", query).replace("{context}", context)
print(prompt)

"""
已知信息：
ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考我们的博客。

为了方便下游开发者针对自己的应用场景定制模型，我们同时实现了基于 P-Tuning v2 的高效参数微调方法 (使用指南) ，INT4 量化级别下最低只需 7GB 显存即可启动微调。

想让 ChatGLM-6B 更符合你的应用场景？欢迎参与 Badcase 反馈计划。
更强大的性能：基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，评测结果显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。
更长的上下文：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练，允许更多轮次的对话。但当前版本的 ChatGLM2-6B 对单轮超长文档的理解能力有限，我们会在后续迭代升级中着重进行优化。
更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。
更多信息参见 ChatGLM2-6B。 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：chatglm-6b是什么？
"""
```

将查询出的内容和query以及基本模板进行整合。

## 根据prompt用模型生成结果

首先我们使用原始的模型直接对query进行预测：

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, query, history=[])
print(response)
"""
ChatGLM-6B 是一个基于清华大学 KEG 实验室和智谱 AI 公司于 2023 年共同训练的语言模型 GLM-6B 开发的人工智能助手。
"""
```

然后是使用带有知识的提问：

```python
response, history = model.chat(tokenizer, prompt, history=[])
print(response)
"""
ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署(INT4 量化级别下最低只需 6GB 显存)。ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。
"""
```

## 总结

到这里，一个简单的基于知识的问道就全部完成了。

当然，prompt的构建和模型的使用我们并没有和langchain进行整合，整合的难度也不是很大，可参考已经写好的文章。