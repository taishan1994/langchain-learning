在local_doc_qa.py中：

```python
def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):
        vector_store = load_vector_store(vs_path, self.embeddings)
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        torch_gc()
        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query

        for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history
```

前面我们已经讲过怎么用query查询出相关的文本，怎么根据query很查询出的文本构建prompt，接下来将介绍怎么根据prompt生成结果。

这里的llm继承了BaseAnswer，

```python
class BaseAnswer(ABC):
    """上层业务包装器.用于结果生成统一api调用"""

    @property
    @abstractmethod
    def _check_point(self) -> LoaderCheckPoint:
        """Return _check_point of llm."""

    @property
    @abstractmethod
    def _history_len(self) -> int:
        """Return _history_len of llm."""

    @abstractmethod
    def set_history_len(self, history_len: int) -> None:
        """Return _history_len of llm."""

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):
        pass
```

需要实现三个抽象方法。

- _check_point：加载模型相关。
- _history_len：历史的文本长度。
- set_history_len：设置历史的文本长度。

还有generatorAnswer，主要是生成结果相关。

我们以chatglm_llm.py为例：

```python
from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult)


class ChatGLM(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"__call:{prompt}")
        response, _ = self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature
        )
        print(f"response:{response}")
        print(f"+++++++++++++++++++++++++++++++++++")
        return response

    def generatorAnswer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):

        if streaming:
            history += [[]]
            for inum, (stream_resp, _) in enumerate(self.checkPoint.model.stream_chat(
                    self.checkPoint.tokenizer,
                    prompt,
                    history=history[-self.history_len:-1] if self.history_len > 1 else [],
                    max_length=self.max_token,
                    temperature=self.temperature
            )):
                # self.checkPoint.clear_torch_cache()
                history[-1] = [prompt, stream_resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
                yield answer_result
        else:
            response, _ = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token,
                temperature=self.temperature
            )
            self.checkPoint.clear_torch_cache()
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
            yield answer_result
```

ChatGLM继承了BaseAnswer，需要实现上述三个抽象方法。

我们主要看generatorAnswer里面的逻辑。

根据是否是流式输出分别调用模型的stream_chat方法或者chat方法。

在configs/model_config.py中，

```python
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": None,
        "provides": "ChatGLM"
    }
    
# LLM 名称
LLM_MODEL = "chatglm-6b"
```

在models/shared.py中：

```python
import sys
from typing import Any
from models.loader.args import parser
from models.loader import LoaderCheckPoint
from configs.model_config import (llm_model_dict, LLM_MODEL)
from models.base import BaseAnswer

loaderCheckPoint: LoaderCheckPoint = None


def loaderLLM(llm_model: str = None, no_remote_model: bool = False, use_ptuning_v2: bool = False) -> Any:
    """
    init llm_model_ins LLM
    :param llm_model: model_name
    :param no_remote_model:  remote in the model on loader checkpoint, if your load local model to add the ` --no-remote-model
    :param use_ptuning_v2: Use p-tuning-v2 PrefixEncoder
    :return:
    """
    pre_model_name = loaderCheckPoint.model_name
    llm_model_info = llm_model_dict[pre_model_name]

    if no_remote_model:
        loaderCheckPoint.no_remote_model = no_remote_model
    if use_ptuning_v2:
        loaderCheckPoint.use_ptuning_v2 = use_ptuning_v2

    # 如果指定了参数，则使用参数的配置
    if llm_model:
        llm_model_info = llm_model_dict[llm_model]

    loaderCheckPoint.model_name = llm_model_info['name']
    loaderCheckPoint.pretrained_model_name = llm_model_info['pretrained_model_name']

    loaderCheckPoint.model_path = llm_model_info["local_model_path"]

    if 'FastChatOpenAILLM' in llm_model_info["provides"]:
        loaderCheckPoint.unload_model()
    else:
        loaderCheckPoint.reload_model()

    provides_class = getattr(sys.modules['models'], llm_model_info['provides'])
    modelInsLLM = provides_class(checkPoint=loaderCheckPoint)
    if 'FastChatOpenAILLM' in llm_model_info["provides"]:
        modelInsLLM.set_api_base_url(llm_model_info['api_base_url'])
        modelInsLLM.call_model_name(llm_model_info['name'])
        modelInsLLM.set_api_key(llm_model_info['api_key'])
    return modelInsLLM
```

这里注意loaderCheckPoint.reload_model()，因为我们使用的是chatglm-6b，所以调用这行代码。

- python中的sys.modules是一个全局字典，从python程序启动就加载到了内存，用于保存当前已导入(加载)的所有模块名和模块对象。
- **getattr()** 函数用于返回一个对象属性值。

得到的provides_class是ChatGLM的一个实例，其需要传入一个loaderCheckPoint，最后返回该实例。

然后看到self.checkPoint.model.，其实际上就是从transformers加载的chatglm，然后调用其chat或stream_chat方法。

这里我们去hugging face上找到其代码：https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py

```python
@torch.no_grad()
def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, num_beams=1,
         do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                  "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    inputs = self.build_inputs(tokenizer, query, history=history)
    outputs = self.generate(**inputs, **gen_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = self.process_response(response)
    history = history + [(query, response)]
    return response, history

@torch.no_grad()
def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values=None,
                max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
                return_past_key_values=False, **kwargs):
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                  "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if past_key_values is None and not return_past_key_values:
        inputs = self.build_inputs(tokenizer, query, history=history)
    else:
        inputs = self.build_stream_inputs(tokenizer, query, history=history)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0]
        if self.transformer.pre_seq_len is not None:
            past_length -= self.transformer.pre_seq_len
        inputs.position_ids += past_length
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
        inputs['attention_mask'] = attention_mask
    for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                        return_past_key_values=return_past_key_values, **gen_kwargs):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        if response and response[-1] != "�":
            response = self.process_response(response)
            new_history = history + [(query, response)]
            if return_past_key_values:
                yield response, new_history, past_key_values
            else:
                yield response, new_history

@torch.no_grad()
def stream_generate(
        self,
        input_ids,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        return_past_key_values=False,
        **kwargs,
):
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

    if generation_config is None:
        generation_config = self.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = self._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
        if return_past_key_values:
            yield input_ids, outputs.past_key_values
        else:
            yield input_ids
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break
```

回到最开始的，输出的结果被封装为AnswerResult()，最后返回。

```python
for answer_result in self.llm.generatorAnswer(prompt=prompt, history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history
```

到这里，怎么加载模型并使用prompt进行预测就全部完成了。
