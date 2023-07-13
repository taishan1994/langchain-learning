在chains下的loca_doc_qa.py

```python
related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
torch_gc()

if len(related_docs_with_score) > 0:
    prompt = generate_prompt(related_docs_with_score, query)
else:
    prompt = query

```

之前我们已经讲过根据query查询相关docs的原理，查询出来的related_docs_with_score是一个列表，列表里面是一个元组，(Document, 分数)。

如果查询出结果，则generate_prompt处理，否则prompt就是query，直接让模型根据query进行生成结果。

看看generate_prompt()函数：

```python
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    # 将查询出的文本进行拼接
    context = "\n".join([doc.page_content for doc in related_docs])
    # 填充到prompt里面
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt
```

最后直接返回prompt即可。

