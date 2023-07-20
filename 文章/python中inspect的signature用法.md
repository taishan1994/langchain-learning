用法：

`inspect.signature(callable, *, follow_wrapped=True, globals=None, locals=None, eval_str=False)`

返回给定 `callable` 的 [`Signature`](https://vimsky.com/cache/index.php?source=https%3A//docs.python.org/3/library/inspect.html%23inspect.Signature) 对象：

看一个具体的例子：

```python
>>> from inspect import signature
>>> def foo(a, *, b:int, **kwargs):
...     pass

>>> sig = signature(foo)

>>> str(sig)
'(a, *, b:int, **kwargs)'

>>> str(sig.parameters['b'])
'b:int'

>>> sig.parameters['b'].annotation
<class 'int'>
```

接受广泛的 Python 可调用对象，从普通函数和类到 [`functools.partial()`](https://vimsky.com/examples/usage/python-functools.partial-py.html) 对象。

functools.partial()我们之前已经讲解过。