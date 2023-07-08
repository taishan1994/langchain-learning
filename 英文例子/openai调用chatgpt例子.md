```python
import openai
openai.api_key = "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"

"""用于打印模型列表
models = openai.Model.list()
total = len(models.data)
for i in range(total):
  print(models.data[i].id)
"""

# create a chat completion
def get_completion(prompt, model_name="gpt-3.5-turbo"):
  messages = [{"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
      model=model_name,
      messages=messages, 
      stream=False, 
      temperature=0)
  print(response)
  return response.choices[0].message["content"]


text = "I don't know who are you"

prompt = f"""{text} 翻译为中文。"""
print(get_completion(prompt).encode("utf-8").decode("utf-8"))
```

