## 基本例子

```python
from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma

names_and_descriptions = [
    ("physics", ["for questions about physics"]),
    ("math", ["for questions about math"]),
]

router_chain = EmbeddingRouterChain.from_names_and_descriptions(
    names_and_descriptions, Chroma, CohereEmbeddings(), routing_keys=["input"]
)

    Using embedded DuckDB without persistence: data will be transient

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

print(chain.run("What is black body radiation?"))

    
    
    > Entering new MultiPromptChain chain...
    physics: {'input': 'What is black body radiation?'}
    > Finished chain.
    
    
    Black body radiation is the emission of energy from an idealized physical body (known as a black body) that is in thermal equilibrium with its environment. It is emitted in a characteristic pattern of frequencies known as a black-body spectrum, which depends only on the temperature of the body. The study of black body radiation is an important part of astrophysics and atmospheric physics, as the thermal radiation emitted by stars and planets can often be approximated as black body radiation.


print(
    chain.run(
        "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3"
    )
)


    
    
    > Entering new MultiPromptChain chain...
    math: {'input': 'What is the first prime number greater than 40 such that one plus the prime number is divisible by 3'}
    > Finished chain.
    ?
    
    Answer: The first prime number greater than 40 such that one plus the prime number is divisible by 3 is 43.

```

## EmbeddingRouterChain

继承了RouterChain，这里不作展开，之前已讲解过。

```python
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.router.base import RouterChain
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


class EmbeddingRouterChain(RouterChain):
    """Class that uses embeddings to route between options."""

    vectorstore: VectorStore
    routing_keys: List[str] = ["query"]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the LLM chain prompt expects.

        :meta private:
        """
        return self.routing_keys

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _input = ", ".join([inputs[k] for k in self.routing_keys])
        results = self.vectorstore.similarity_search(_input, k=1)
        return {"next_inputs": inputs, "destination": results[0].metadata["name"]}

    @classmethod
    def from_names_and_descriptions(
        cls,
        names_and_descriptions: Sequence[Tuple[str, Sequence[str]]],
        vectorstore_cls: Type[VectorStore],
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> EmbeddingRouterChain:
        """Convenience constructor."""
        documents = []
        for name, descriptions in names_and_descriptions:
            for description in descriptions:
                documents.append(
                    Document(page_content=description, metadata={"name": name})
                )
        vectorstore = vectorstore_cls.from_documents(documents, embeddings)
        return cls(vectorstore=vectorstore, **kwargs)
```

调用from_names_and_descriptions将其它链的名称和描述封装为Document。然后转换为一个vectorstore，最后对类的属性进行初始化。

然后传入到：

```python
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
```

和路由链不同，先要根据_call里面通过向量来获取下一个目标链的名称和描述，其余的应该是基本一致的。