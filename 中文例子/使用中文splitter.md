之前已经讲解过了ChineseTextSplitter后面的原理，可以找到相关文章进行查看。

这里将langchian-chatglm里面的拿出来讲讲，先看代码：

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
sentence_size = 100
loader = TextLoader(filepath, autodetect_encoding=True)
textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
docs = loader.load_and_split(textsplitter)

from pprint import pprint

pprint(docs)

“”“
Document(page_content='本节开始后不久，奇才两度三分命中，而塞拉芬此后又空中接力扣篮，奇才很快就反超比分。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='本节还有3分57秒时，杰弗森上篮后，爵士将比分追成36-39，沃尔马上突破上篮还以一球，克劳福德此后也中投命中，奇才又将差距拉开。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='本节奇才以35-22大胜，上半场以51-43领先。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='易建联在第三节终于首次得分，本节开始后不久，他抢下进攻篮板，直接补篮得手，奇才取得10分的优势。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='本节后半段易建联又两度投篮命中，本节还有2分25秒时，易建联中投得手后，奇才以70-61领先，马丁紧接着三分得手，奇才又取得两位数的优势。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='易建联本节得了6分，三节过后，奇才以73-63领先。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='第四节易建联只打了3分钟就6次犯规下场，他还没来得及出手投篮，只抢下一个篮板。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='奇才遭到爵士顽强的反击，在本节还有1分30秒时，埃文斯上篮得手，爵士只以82-83落后。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='沃尔关键时刻的投篮被盖帽，埃文斯两罚两中后，爵士在本节结束前45.', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='1秒以84-83反超，这是他们下半场首度领先。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='第四节还有7分36秒时起，奇才在7分32秒内未能投中一球，在这段时间只得了1分。', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
 Document(page_content='爵士领先2分后，克劳福德终于在本节还有4.', metadata={'source': '/content/drive/MyDrive/lm_pretrained/data/test_corpus.txt'}),
“”“
```

当然，我们也可以根据自己的需求修改split_text里面的逻辑。