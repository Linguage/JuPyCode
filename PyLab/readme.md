# Python基本操作命令指南

## 基础语法

### 变量与数据类型
```python
# 变量赋值
x = 10                  # 整数
y = 3.14                # 浮点数
name = "Python"         # 字符串
is_valid = True         # 布尔值
my_list = [1, 2, 3]     # 列表
my_tuple = (1, 2, 3)    # 元组
my_dict = {"a": 1, "b": 2}  # 字典
my_set = {1, 2, 3}      # 集合

# 查看变量类型
print(type(x))          # <class 'int'>
```

### 基本运算
```python
# 算术运算符
print(10 + 5)           # 加法: 15
print(10 - 5)           # 减法: 5
print(10 * 5)           # 乘法: 50
print(10 / 5)           # 除法: 2.0
print(10 // 3)          # 整除: 3
print(10 % 3)           # 取余: 1
print(10 ** 2)          # 幂运算: 100

# 比较运算符
print(10 > 5)           # True
print(10 < 5)           # False
print(10 == 5)          # False
print(10 != 5)          # True
```

### 字符串操作
```python
# 字符串创建与连接
s1 = "Hello"
s2 = 'World'
s3 = s1 + " " + s2      # "Hello World"

# 字符串格式化
name = "张三"
age = 25
# f-string (Python 3.6+)
print(f"姓名: {name}, 年龄: {age}")
# format方法
print("姓名: {}, 年龄: {}".format(name, age))

# 字符串切片
s = "Python编程"
print(s[0:2])           # "Py"
print(s[2:])            # "thon编程"
print(s[::-1])          # "程编nohtyP" (反转)
```

## 控制流

### 条件语句
```python
x = 10
if x > 10:
    print("x大于10")
elif x == 10:
    print("x等于10")
else:
    print("x小于10")
```

### 循环语句
```python
# for循环
for i in range(5):
    print(i)            # 打印0到4

# while循环
i = 0
while i < 5:
    print(i)
    i += 1

# break和continue
for i in range(10):
    if i == 3:
        continue        # 跳过当前循环
    if i == 8:
        break           # 结束整个循环
    print(i)
```

## 数据结构

### 列表操作
```python
# 创建列表
fruits = ["苹果", "香蕉", "橙子"]

# 访问元素
print(fruits[0])        # 第一个元素: "苹果"
print(fruits[-1])       # 最后一个元素: "橙子"

# 添加元素
fruits.append("葡萄")    # 在末尾添加
fruits.insert(1, "梨")   # 在指定位置插入

# 删除元素
fruits.remove("香蕉")    # 移除指定元素
del fruits[0]           # 删除指定位置元素
popped = fruits.pop()   # 弹出最后一个元素并返回

# 列表推导式
squares = [x**2 for x in range(10)]
```

### 字典操作
```python
# 创建字典
student = {
    "name": "李四", 
    "age": 20,
    "score": 95
}

# 访问元素
print(student["name"])
print(student.get("age"))  # 安全获取

# 添加/修改元素
student["gender"] = "男"
student["age"] = 21

# 删除元素
del student["score"]
phone = student.pop("phone", "未设置")  # 带默认值的删除

# 遍历字典
for key in student:
    print(f"{key}: {student[key]}")

for key, value in student.items():
    print(f"{key}: {value}")
```

## 函数

### 函数定义与调用
```python
# 基本函数
def greet(name):
    return f"你好，{name}！"

print(greet("王五"))

# 默认参数
def greet_with_time(name, time="早上"):
    return f"{time}好，{name}！"

print(greet_with_time("王五"))
print(greet_with_time("王五", "下午"))

# 不定长参数
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))  # 15
```

### Lambda表达式
```python
# 简单匿名函数
square = lambda x: x**2
print(square(5))  # 25

# 结合内置函数使用
numbers = [1, 4, 2, 8, 5]
sorted_numbers = sorted(numbers, key=lambda x: x)
```

## 文件操作

### 读写文件
```python
# 写入文件
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("这是第一行\n")
    f.write("这是第二行\n")

# 读取文件
with open("example.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)

# 逐行读取
with open("example.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())
```

## 异常处理

### try-except结构
```python
try:
    x = int(input("请输入一个数字: "))
    result = 10 / x
    print(f"结果是: {result}")
except ValueError:
    print("输入不是有效的数字")
except ZeroDivisionError:
    print("不能除以零")
except Exception as e:
    print(f"发生了错误: {e}")
finally:
    print("这部分代码总是会执行")
```

## 模块和包

### 导入模块
```python
# 导入标准库
import math
print(math.sqrt(16))  # 4.0

# 导入特定函数
from random import randint
print(randint(1, 10))  # 1到10之间的随机数

# 重命名导入
import numpy as np
arr = np.array([1, 2, 3])
```

### 常用标准库
```python
# 日期时间
import datetime
now = datetime.datetime.now()
print(f"当前时间: {now}")

# 正则表达式
import re
pattern = r"\d+"
result = re.findall(pattern, "我有10个苹果和20个香蕉")
print(result)  # ['10', '20']

# JSON处理
import json
data = {"name": "Python", "version": 3.9}
json_str = json.dumps(data)
print(json_str)
```

## 代码优化与调试技巧

### 性能测试
```python
import time

start_time = time.time()
# 要测试的代码
for i in range(1000000):
    pass
end_time = time.time()
print(f"运行时间: {end_time - start_time} 秒")
```

### 调试打印
```python
# 格式化打印
import pprint
complex_data = {"users": [{"name": "张三", "age": 25}, {"name": "李四", "age": 30}]}
pprint.pprint(complex_data)

# 调试信息
def calculate(a, b):
    print(f"DEBUG: a={a}, b={b}")
    result = a * b
    print(f"DEBUG: result={result}")
    return result
```

