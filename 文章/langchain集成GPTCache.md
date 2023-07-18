一般的，针对于一些问的比较频繁的问题，我们可以将其缓存下来，以后如果遇到重复的问题，直接从缓存中读取，可以大大加快反应的速度。在langchain已经有原生的缓存功能，比如：

```python
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

// CPU times: user 14.2 ms, sys: 4.9 ms, total: 19.1 ms
// Wall time: 1.1 s
llm("Tell me a joke")

// CPU times: user 162 µs, sys: 7 µs, total: 169 µs
// Wall time: 175 µs
llm("Tell me a joke")

```

LangChain 命中缓存的条件是两个问题必须完全相同。但是在实际使用中，这种情况十分罕见，因此很难命中缓存。这也意味着，我们还有很多空间可以用来提升缓存利用率，集成 GPTCache 就是方法之一。

GPTCache 首先将输入的问题转化为 embedding 向量，随后 GPTCache 会在缓存中进行向量近似搜索。获取[向量相似性](https://www.zhihu.com/search?q=向量相似性&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3122732822})检索的结果后，GPTCache 会执行相似性评估，并将达到设置阈值的结果作为最终返回结果。大家可以通过调整阈值来调节 GPTCache 模糊搜索结果的准确性。

以下示例中在 LangChain 中集成了 GPTCache，并使用了 GPTCache 进行向量相似性检索。

```python
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from langchain.cache import GPTCache
import hashlib
def get_hashed_name(name):
   return hashlib.sha256(name.encode()).hexdigest()
def init_gptcache(cache_obj: Cache, llm: str):
   hashed_llm = get_hashed_name(llm)
   init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")
langchain.llm_cache = GPTCache(init_gptcache)

# The first time, it is not yet in cache, so it should take longer
# CPU times: user 1.42 s, sys: 279 ms, total: 1.7 s
# Wall time: 8.44 s
llm("Tell me a joke")

# This is an exact match, so it finds it in the cache
# CPU times: user 866 ms, sys: 20 ms, total: 886 ms
# Wall time: 226 ms
llm("Tell me a joke")

# This is not an exact match, but semantically within distance so it hits!
# CPU times: user 853 ms, sys: 14.8 ms, total: 868 ms
# Wall time: 224 ms
llm("Tell me joke")
```

作者：Zilliz
链接：https://www.zhihu.com/question/606436913/answer/3122732822
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。