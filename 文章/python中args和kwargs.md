- *args 表示任何多个无名参数， 他本质上是一个 tuple。
- ** kwargs 表示关键字参数， 它本质上是一个 dict。
- 同时使用时必须要求 *args 参数列要在** kwargs 前面。

直接看一些例子：

```python
a = [1,2,3]
b = [*a,4,5,6]
b 
# ----------------- 输出结果 -----------------
# [1, 2, 3, 4, 5, 6]
# ----------------- 总结 -----------------
# 将a的内容移入（解包）到新列表b中。

def print_func(*args):
    print(type(args))
    print(args)
print_func(1,2,'python希望社',[])

"""
<class 'tuple'>
(1, 2, 'python希望社', [])
"""

def print_func(x,y,*args):
    print(type(x))
    print(x)
    print(y)
    print(type(args))
    print(args)
print_func(1,2,'python希望社',[])

"""
<class 'int'>
1
2
<class 'tuple'>
('python希望社', [])
"""

def print_func(**kwargs):
    print(type(kwargs))
    print(kwargs)

print_func(a=1, b=2, c='呵呵哒', d=[])

"""
<class 'dict'>
{'a': 1, 'b': 2, 'c': '呵呵哒', 'd': []}
"""

def print_func(x, *args, **kwargs):
    print(x)
    print(args)
    print(kwargs)

print_func(1, 2, 3, 4, y=1, a=2, b=3, c=4)
"""
1
(2, 3, 4)
{'y': 1, 'a': 2, 'b': 3, 'c': 4}

"""
```

到这，你基本上就了解它们是什么了。

以上内容摘自：[(29条消息) 【Python】`*args` 和 `**kwargs`的用法【最全详解】_春风惹人醉的博客-CSDN博客](https://blog.csdn.net/GODSuner/article/details/117961990)