在local_doc_qa.py中使用到：

```python
loader = TextLoader(filepath, autodetect_encoding=True)
textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
docs = loader.load_and_split(textsplitter)
```

ChineseTextSplitter继承了CharacterTextSplitter。

```python
class CharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters."""

    def __init__(self, separator: str = "\n\n", **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        if self._separator:
            splits = text.split(self._separator)
        else:
            splits = list(text)
        return self._merge_splits(splits, self._separator)
```

CharacterTextSplitter继承了TextSplitter，看看TextSplitter：

```python
def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

class TextSplitter(ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                documents.append(Document(page_content=chunk, metadata=_metadatas[i]))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len >= self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len > self._chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0])
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
        """Text splitter that uses HuggingFace tokenizer to count length."""
        try:
            from transformers import PreTrainedTokenizerBase

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "Tokenizer received was not an instance of PreTrainedTokenizerBase"
                )

            def _huggingface_tokenizer_length(text: str) -> int:
                return len(tokenizer.encode(text))

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please it install it with `pip install transformers`."
            )
        return cls(length_function=_huggingface_tokenizer_length, **kwargs)

    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "gpt2",
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
        **kwargs: Any,
    ) -> TextSplitter:
        """Text splitter that uses tiktoken encoder to count length."""
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate max_tokens_for_prompt. "
                "Please it install it with `pip install tiktoken`."
            )

        # create a GPT-3 encoder instance
        enc = tiktoken.get_encoding(encoding_name)

        def _tiktoken_encoder(text: str, **kwargs: Any) -> int:
            return len(
                enc.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                    **kwargs,
                )
            )

        return cls(length_function=_tiktoken_encoder, **kwargs)
```

唯一一个抽象方法需要我们去实现：split_text。

回到CharacterTextSplitter，其实现的split_text就是按照指定的分隔符将文本进行划分，然后用父类的_merge_splits将文本进行合并，主要是根据chunk_size和chunk_overlap进行合并。

而在ChineseTextSplitter中，

```python
from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
from configs.model_config import SENTENCE_SIZE


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = SENTENCE_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

    def split_text(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls
```

各种各种的规则，拿个例子来看看：

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import re
from typing import List



class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = None, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

    def split_text(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls

filepath = "/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt"
sentence_size = 20
loader = TextLoader(filepath, autodetect_encoding=True)
textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
docs = loader.load_and_split(textsplitter)

from pprint import pprint
pprint(docs)

"""
[Document(page_content='又一次上架了，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='这次比上次还激动，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='甚至激动到了上传了章节却不知道发出来的地步。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='尴尬，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='关于新书，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='上架前成绩好得有些出乎土豆的意料，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='对于这份厚硕的成绩，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='土豆心里还真有几分惶恐与忐忑，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='虽说曾经有人说土豆是刷出来的数据，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='对于这些留言，', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
......
"""
```

会把结果分得很细。



