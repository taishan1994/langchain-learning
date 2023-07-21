# llama_generative_agent.ipynb

从这里面看起一个个来看：

## LlamaCpp

```python
from langchain.llms import LlamaCpp

# local_path = "/Users/jon/Documents/models/stable-vicuna-13B.ggml.q5_1.bin"
local_path = "/Users/jon/Downloads/ggml-vicuna-13b-1.1-q4_2.bin"
llm = LlamaCpp(
    model_path=local_path, verbose=True, n_batch=256, temperature=0.3, n_ctx=2048,
                    use_mmap=False, stop=["###"]
)
```

LlamaCpp中的self.client实际上就是：

```python
from llama_cpp import Llama
values["client"] = Llama(model_path, **model_params)

```

## LlamaCppEmbeddings

```python
from llama_cpp import Llama
values["client"] = Llama(model_path, **model_params)
```

```python
from langchain.embeddings import LlamaCppEmbeddings
# 实际把文本编码成向量的还是self.client.embed(text)
# 可以借鉴这个来自定义自己的向量转换模块
```

```python
from retrivers.llama_time_weighted_retriever import LlamaTimeWeightedVectorStoreRetriever
```

## LlamaTimeWeightedVectorStoreRetriever 

LlamaTimeWeightedVectorStoreRetriever这个是自定义的。在retrivers下。

```python
from copy import deepcopy
from datetime import datetime
from typing import Any, List, Optional

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document


def _get_hours_passed(time: datetime, ref_time: datetime | str) -> float:
    """Get the hours passed between two datetime objects."""
    if isinstance(ref_time, str):
        ref_time = datetime.fromisoformat(ref_time)
    return (time - ref_time).total_seconds() / 3600


class LlamaTimeWeightedVectorStoreRetriever(TimeWeightedVectorStoreRetriever):
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        current_time = kwargs.get("current_time", datetime.now())
        # Avoid mutating input documents
        dup_docs = [deepcopy(d) for d in documents]
        for i, doc in enumerate(dup_docs):
            if "last_accessed_at" not in doc.metadata:
                doc.metadata["last_accessed_at"] = str(current_time)
            if "created_at" not in doc.metadata:
                doc.metadata["created_at"] = str(current_time)
            doc.metadata["buffer_idx"] = len(self.memory_stream) + i
        self.memory_stream.extend(dup_docs)
        return self.vectorstore.add_documents(dup_docs, **kwargs)

    def _get_combined_score(
        self,
        document: Document,
        vector_relevance: Optional[float],
        current_time: datetime,
    ) -> float:
        """Return the combined score for a document."""
        hours_passed = _get_hours_passed(
            current_time,
            document.metadata["last_accessed_at"],
        )
        score = (1.0 - self.decay_rate) ** hours_passed
        for key in self.other_score_keys:
            if key in document.metadata:
                score += document.metadata[key]
        if vector_relevance is not None:
            score += vector_relevance
        return score
```

也很好理解：

- add_documents：给document设置一些元学习
- 根据分数进行加权计算。

## EnhancedChroma

```python
from vectorestores.chroma import EnhancedChroma
```

这个也是自定义的，在vectorestores下。

```python
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
from chromadb.errors import NoIndexException
from langchain.schema import Document
from langchain.vectorstores import Chroma

logger = logging.getLogger(__name__)


def default_relevance_score_fn(score: float):
    import math

    return 1 / (1 + math.exp(-score / 100000)) - 0.5


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class EnhancedChroma(Chroma):
    def __init__(
        self,
        relevance_score_fn: Callable[[float], float] = default_relevance_score_fn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.relevance_score_fn = relevance_score_fn

    def _similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores on a scale from 0 to 1."""
        if self.relevance_score_fn is None:
            raise ValueError("a relevance_score_fn is required.")
        try:
            docs_and_scores = self.similarity_search_with_score(query, k=k)
            return [
                (doc, self.relevance_score_fn(score)) for doc, score in docs_and_scores
            ]
        except NoIndexException:
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query
                text with distance in float.
        """
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query], n_results=k, where=filter
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding], n_results=k, where=filter
            )

        return _results_to_docs_and_scores(results)

    def __query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """Query the chroma collection."""
        for i in range(n_results, 0, -1):
            try:
                return self._collection.query(
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    n_results=i,
                    where=where,
                )
            except chromadb.errors.NotEnoughElementsException:
                logger.warning(
                    f"Chroma collection {self._collection.name} "
                    f"contains fewer than {i} elements."
                )
        return []
```

它有一段测试的代码：

```python
import pytest
from chroma import EnhancedChroma
from langchain.embeddings import LlamaCppEmbeddings


def test():
    local_path = "/Users/jon/Downloads/ggml-vicuna-13b-1.1-q4_2.bin"
    embeddings = LlamaCppEmbeddings(model_path=local_path)

    vs = EnhancedChroma.from_texts([], embedding=embeddings)

    docs = vs.similarity_search_with_score("how does tommie feel?", k=1)
    print(docs)


def test_default_relevance_score_fn():
    print(default_relevance_score_fn(14000.0))
    print(default_relevance_score_fn(0))
    print(default_relevance_score_fn(20000.0))
    print(default_relevance_score_fn(200000.0))
    print(default_relevance_score_fn(2000000.0))
```

from_texts在langchain/vectorstores/chroma.py里面，实际上先初始化Chroma的一些属性，然后调用add_texts方法。这里面就是对document进行向量化并加入到向量库中，根据使用的向量库不同会有一些差异。

```python
def create_new_memory_retriever():
    embeddings_model = LlamaCppEmbeddings(model_path=local_path)
    vs = EnhancedChroma(embedding_function=embeddings_model)
    return LlamaTimeWeightedVectorStoreRetriever(vectorstore=vs, other_score_keys=["importance"], k=15)
```

我们可以根据上述方法自定义自己的检索器。

## LlamaGenerativeAgent

```python
from generative_agents.llama_generative_agent import LlamaGenerativeAgent
```

LlamaGenerativeAgent也是自定义的，在generative_agents下。

```python
from datetime import datetime
from typing import Dict, Any

from langchain import PromptTemplate, LLMChain
from langchain.experimental.generative_agents.generative_agent import \
    GenerativeAgent


class LlamaGenerativeAgent(GenerativeAgent):

    system_prompt: str = (
        "A chat between a curious user and an artificial intelligence assistant. The assistant "
        "gives helpful, detailed, and polite answers to the user's questions.\n"
        "###USER: %s\n"
        "###ASSISTANT: ")

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )

    def _get_entity_from_observation(self, observation: str) -> str:
        # TODO: better prompt for conversations.
        instruction = (
            f"Extract the entity from the following observation without explanation.\n"
            f"Observation: {observation}\n"
        )
        prompt = PromptTemplate.from_template(
            self.system_prompt % instruction
        )
        return self.chain(prompt).run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        instruction = (
            f"What is the {entity_name} doing in the following observation?\n"
            f"Observation: {observation}\n"
        )
        prompt = PromptTemplate.from_template(
            self.system_prompt % instruction
        )
        return (
            self.chain(prompt).run(entity=entity_name, observation=observation).strip()
        )

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            "{q1}?\n"
            "Context from memory:\n"
            "{relevant_memories}\n"
            "Relevant context:"
        )
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} is {entity_action}"
        return self.chain(prompt=prompt).run(q1=q1, queries=[q1, q2]).strip()

    def summarize_speaker_memories(self, speaker: str, observation: str) -> str:
        instruction = (
            f"what is the most possible relationship between {self.name} and {speaker} in the"
            f" following observation? Do not embellish if you don't know. Do not return a list.\n"
            "Observation: {relevant_memories}\n"
        )
        prompt = PromptTemplate.from_template(
            self.system_prompt % instruction
        )
        return self.chain(prompt=prompt).run(me=self.name, speaker=speaker, queries=[f"{speaker}"]).strip()

    def _compute_agent_summary(self) -> str:
        instruction = (
            f"Summarize {self.name}'s core characteristics given the following input. Do not "
            f"embellish if you don't know. Do not return a list.\n"
            "Input: {relevant_memories}\n"
        )
        prompt = PromptTemplate.from_template(
            self.system_prompt % instruction
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )

    def _generate_dialogue_reaction(self, speaker: str, observation: str, suffix: str) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_speaker_memories(speaker, observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation= speaker + " says " + observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def generate_dialogue(self, speaker: str, observation: str):
        """React to a given observation."""
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_dialogue_reaction(
            speaker,
            observation,
            call_to_action_template
        )
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = self._clean_response(result.split("GOODBYE:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{observation} and said {farewell}"
                },
            )
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = self._clean_response(result.split("SAY:")[-1])
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                                                f"{observation} and said {response_text}"
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result
```

其继承了GenerativeAgent，`from langchain.experimental.generative_agents.generative_agent import GenerativeAgent`  主要是继承了GenerativeAgent的一些属性和方法，具体使用到的时候再看具体的就好。

## LlamaGenerativeAgentMemory

```python
from generative_agents.llama_memory import LlamaGenerativeAgentMemory
```

```python
import logging
import re
from datetime import datetime
from typing import List

from langchain import PromptTemplate
from langchain.experimental.generative_agents.memory import \
    GenerativeAgentMemory
from langchain.schema import Document

logger = logging.getLogger(__name__)


class LlamaGenerativeAgentMemory(GenerativeAgentMemory):
    def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        template = (
            "On the scale of 1 to 10, where 1 is not important at all"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely important (e.g., a break up, college"
            + " acceptance), rate the importance of the"
            + " following piece of memory. You must respond with a single integer."
            + "\nMemory: I met my wife Jane when I was 23"
            + "\nRating: 9"
            "\nMemory: I visited Italy in 2020"
            "\nRating: 5"
            "\nMemory: {memory_content}"
            "\nRating: "
        )
        prompt = PromptTemplate.from_template(template)

        score = self.chain(prompt).run(memory_content=memory_content).strip()

        logger.warning(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(score[0]) / 10) * self.importance_weight
        else:
            return 0.0

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content in content_strs:
                continue
            content_strs.add(mem.page_content)
            created_time = datetime.fromisoformat(mem.metadata["created_at"]).strftime(
                "%B %d, %Y, %I:%M %p"
            )
            content.append(f"- {created_time}: {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])
```

还是以用到什么看什么的原则，接着往下看。

## 开始使用

```python
tommies_memory = LlamaGenerativeAgentMemory(
    llm=llm,
    memory_retriever=create_new_memory_retriever(),
    reflection_threshold=8, # we will give this a relatively low number to show how reflection works
    verbose=True,
)

tommie = LlamaGenerativeAgent(
    name="Tommie",
    age=25,
    traits="anxious, likes design, talkative", # You can add more persistent traits here
    status="looking for a job", # When connected to a virtual world, we can have the characters update their status
    memory_retriever=create_new_memory_retriever(),
    llm=llm,
    memory=tommies_memory,
    verbose=True,
)

```

这里初始化一些属性，接着往下：

```python
print(tommie.get_summary(force_refresh=True))
```

看下LlamaGenerativeAgent对象里面的get_summary方法，实际上是使用父类的GenerativeAgent里面的get_summary方法：

```python
 def get_summary(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a descriptive summary of the agent."""
        current_time = datetime.now() if now is None else now
        # last_refreshed在初始化的时候就赋值了
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary #初始化为空
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        age = self.age if self.age is not None else "N/A"
        return (
            f"Name: {self.name} (age: {age})"
            + f"\nInnate traits: {self.traits}"
            + f"\n{self.summary}"
        )
        
   def _compute_agent_summary(self) -> str:
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )
        
 def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory
        )
```

对name's的core characteristics进行一个总结。

接着看下一个实例：

```python
# We can add memories directly to the memory object
tommie_observations = [
    "Tommie remembers his dog, Bruno, from when he was a kid",
    "Tommie feels tired from driving so far",
    "Tommie sees the new home",
    "The new neighbors have a cat",
    "The road is noisy at night",
    "Tommie is hungry",
    "Tommie tries to get some rest.",
]
for observation in tommie_observations:
    tommie.memory.add_memory(observation)
```

 memory: GenerativeAgentMemory，看下GenerativeAgentMemory

### GenerativeAgentMemory

```python
def add_memory(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        self.aggregate_importance += importance_score
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )
        
        # memory_retriever: TimeWeightedVectorStoreRetriever
        result = self.memory_retriever.add_documents([document], current_time=now)

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # Hack to clear the importance from reflection
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result
    
 def _score_memory_importance(self, memory_content: str) -> float:
        """Score the absolute importance of the given memory."""
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: "
        )
        score = self.chain(prompt).run(memory_content=memory_content).strip()
        if self.verbose:
            logger.info(f"Importance score: {score}")
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0
        
 def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        if self.verbose:
            logger.info("Character is reflecting")
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now)
            for insight in insights:
                self.add_memory(insight, now=now)
            new_insights.extend(insights)
        return new_insights
    
def _get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """Return the 3 most salient high-level questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 3 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line."
        )
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        result = self.chain(prompt).run(observations=observation_str)
        return self._parse_list(result)
    
 def _get_insights_on_topic(
        self, topic: str, now: Optional[datetime] = None
    ) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Statements relevant to: '{topic}'\n"
            "---\n"
            "{related_statements}\n"
            "---\n"
            "What 5 high-level novel insights can you infer from the above statements "
            "that are relevant for answering the following question?\n"
            "Do not include any insights that are not relevant to the question.\n"
            "Do not repeat any insights that have already been made.\n\n"
            "Question: {topic}\n\n"
            "(example format: insight (because of 1, 5, 3))\n"
        )

        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join(
            [
                self._format_memory_detail(memory, prefix=f"{i+1}. ")
                for i, memory in enumerate(related_memories)
            ]
        )
        result = self.chain(prompt).run(
            topic=topic, related_statements=related_statements
        )
        # TODO: Parse the connections between memories and insights
        return self._parse_list(result)
```

在加入了一些记忆的时候，实际上触发了一些self.chain()，比如`_get_topics_of_reflection`和`_get_insights_on_topic`里面。

根据Prompt总结一下：

- 先定义智能体 的一些基本信息，比如：
    ```python
    name="Tommie",
    age=25,
    traits="anxious, likes design, talkative", # You can add more persistent traits here
    status="looking for a job", # When connected to a virtual world, we can have the characters update their status
    ```

- 总结该智能体的一些核心特性
    ```python
    prompt = PromptTemplate.from_template(
                "How would you summarize {name}'s core characteristics given the"
                + " following statements:\n"
                + "{relevant_memories}"
                + "Do not embellish."
                + "\n\nSummary: "
            )
    # The agent seeks to think about their core characteristics.
    return (
        self.chain(prompt)
        .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
        .strip()
    )
    ```

- 添加一些观测值到记忆里面
    ```python
    tommie_observations = [
        "Tommie remembers his dog, Bruno, from when he was a kid",
        "Tommie feels tired from driving so far",
        "Tommie sees the new home",
        "The new neighbors have a cat",
        "The road is noisy at night",
        "Tommie is hungry",
        "Tommie tries to get some rest.",
    ]
    ```

    在调用add_memory方法的时候可能会对添加的观测进行一个评分，并思考观测得到的主体，并给出一些见解。

## generate_reaction和generate_dialogue

```python
# Now that Tommie has 'memories', their self-summary is more descriptive, though still rudimentary.
# We will see how this summary updates after more observations to create a more rich description.
print(tommie.get_summary(force_refresh=True))
print(tommie.generate_reaction("Tommie sees his neighbor's cat"))
print(tommie.generate_dialogue("Dad", "Have you got a new job?"))
```

generate_reaction：

```python
def _generate_reaction(
        self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def generate_reaction(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        # AAA
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}",
                self.memory.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{self.name} said {said_value}"
        else:
            return False, result
```

generate_dialogue：

```python
 def _generate_dialogue_reaction(self, speaker: str, observation: str, suffix: str) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
            + suffix
        )
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_speaker_memories(speaker, observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation= speaker + " says " + observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

def generate_dialogue(self, speaker: str, observation: str):
    """React to a given observation."""
    call_to_action_template = (
        "What would {agent_name} say? To end the conversation, write:"
        ' GOODBYE: "what to say". Otherwise to continue the conversation,'
        ' write: SAY: "what to say next"\n\n'
    )
    full_result = self._generate_dialogue_reaction(
        speaker,
        observation,
        call_to_action_template
    )
    result = full_result.strip().split("\n")[0]
    if "GOODBYE:" in result:
        farewell = self._clean_response(result.split("GOODBYE:")[-1])
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                                            f"{observation} and said {farewell}"
            },
        )
        return False, f"{self.name} said {farewell}"
    if "SAY:" in result:
        response_text = self._clean_response(result.split("SAY:")[-1])
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                                            f"{observation} and said {response_text}"
            },
        )
        return True, f"{self.name} said {response_text}"
    else:
        return False, result
```

