以下内容参考：

https://zhuanlan.zhihu.com/p/376012376

https://link.zhihu.com/?target=https%3A//www.tuicool.com/articles/ymUjuqY

一句话总结：根据现有函数返回一个新的函数，举个简单的例子：

```python
from functools import partial
def addone(x):
    return x+1

add = partial(addone, 3)
print(add())
# 输出4

# 如果有多个参数
def get_info(name="tom", age=18):
    return (name, age)

get_infor = partial(get_info, age=28)
print(get_infor(name="jack"))
```

partial的一个实际作用是**作为回调函数**，假设我们本身的参数就是一个函数名。比如：

```python
def show(name, age):
    print("name {} age {}".format(name, age))
    
def test(callback):
    print("这里做一些事情")
    callback()
    
# 假设我们想用callback调用show呢，怎么传入name和age
# 可以这么改写
def test2(callback, name, age):
    print("这里做一些事情")
    callback(name, age)

# 要是我们还想将callback调用其它的函数呢
# 可以这么写

showp = partial(show, name="tom", age=18)

test(showp)

```

