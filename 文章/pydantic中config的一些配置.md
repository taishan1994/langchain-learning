Pydantic 中的 `Config` 类是用于配置 Pydantic 模型的属性的类。通过在 Pydantic 模型中定义 `Config` 类，可以控制模型的行为，例如序列化和反序列化选项、JSON 解析选项、ORM 集成等等。下面是 Pydantic 中 `Config` 类的常用配置选项：

- `allow_population_by_field_name`: 默认情况下，Pydantic 模型是通过参数名称来构建的。如果将此选项设置为 True，则可以通过参数名称和字段名称来构建模型。
- `anystr_strip_whitespace`: 如果此选项为 True，则将使用 `strip()` 方法从任何字符串类型的值中删除前导和尾随空格。
- `arbitrary_types_allowed`: 如果此选项为 True，则 Pydantic 将接受任何类型的值，并将其视为有效的。否则，只有预定义的类型（例如 str、int、float 等）才会被接受。
- `extra`: 控制是否允许额外的字段。如果设置为 `'ignore'`，则忽略额外的字段；如果设置为 `'allow'`，则接受额外的字段；如果设置为 `'forbid'`，则拒绝额外的字段。
- `json_encoders`: 将自定义编码器添加到 JSON 编码器列表中。
- `json_loads`: 自定义 JSON 解码器。
- `json_dumps`: 自定义 JSON 编码器。
- `keep_untouched`: 如果此选项为 True，则保留原始字典中的任何未修改的字段，而不是在解析后丢弃它们。
- `orm_mode`: 如果此选项为 True，则 Pydantic 将允许从 ORM 模型（例如 SQLAlchemy 模型）中加载数据，并将其转换为适当的类型。默认情况下，Pydantic 假定所有属性都是字符串。
- `validate_assignment`: 如果此选项为 True，则在将值分配给属性时执行验证。如果设置为 False，则不执行验证。
- `use_enum_values`: 如果此选项为 True，则使用枚举值而不是枚举名称来序列化和反序列化枚举类型的属性。默认情况下，Pydantic 使用枚举名称而不是值。

下面是一个示例，展示了如何使用 Pydantic 中的 `Config` 类来配置模型的行为：

```python
from enum import Enum
from typing import List
from pydantic import BaseModel

class Color(str, Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None
    corlor: str

    class Config:
        orm_mode = False

class User(BaseModel):
    id: int
    username: str
    email: str
    password: str
    items: List[Item] = []

    class Config:
        orm_mode = False
        use_enum_values = True
        allow_population_by_field_name = True
        extra = 'ignore'

item = Item(name="python入门", price=18, is_offer=False, corlor=Color.BLUE)
user = User(id=1, username="张三", email="23232323@qq.com", password="123", items=[item])

print(user)

"""
id=1 username='张三' email='23232323@qq.com' password='123' items=[Item(name='python入门', price=18.0, is_offer=False, corlor='blue')]
"""
```

