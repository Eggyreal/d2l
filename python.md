## python 基础用法

### str,list,dict的内置函数
1️⃣str
|方法|作用|示例|
|-------|-------|-------|
|.find(sub)|查找子串 sub 的索引，找不到返回 -1|"hello".find("e") → 1|
|.index(sub)|查找子串 sub 的索引，找不到报错|"hello".index("e") → 1|
|.upper()|转大写|"hello".upper() → "HELLO"|
|.lower()|转小写	|"Hello".lower() → "hello"|
|.replace(old, new)|替换子串|"hello".replace("l","L") → "heLLo"|
|.split(sep)|按分隔符拆分字符串|"a,b,c".split(",") → ['a','b','c']|
|.join(list)|把列表元素拼成字符串|",".join(['a','b','c']) → "a,b,c"|
2️⃣ list
|方法|作用|示例|
|-------|-------|-------|
|.append(x)|在末尾添加元素|[1,2].append(3) → [1,2,3]|
|.extend(list2)|追加另一个列表|[1,2].extend([3,4]) → [1,2,3,4]|
|.insert(i, x)|在索引i插入元素|[1,2].insert(1,9) → [1,9,2]|
|.pop(i)|删除索引 i 元素并返回|[1,2,3].pop(1) → 2, list → [1,3]|
|.remove(x)|删除第一个值为 x 的元素|[1,2,3,2].remove(2) → [1,3,2]|
|.sort()|原地排序|[3,1,2].sort() → [1,2,3]|
|sorted(list)|返回排序后的新列表|sorted([3,1,2]) → [1,2,3]|
3️⃣ dict
|方法|作用|示例|
|-------|-------|-------|
|.keys()|返回所有键|{'a':1}.keys() → dict_keys(['a'])|
|.values()|返回所有值|{'a':1}.values() → dict_values([1])|
|.items()|返回键值对元组|{'a':1}.items() → dict_items([('a',1)])|
|.get(key, default)|获取 key 对应值，不存在返回默认值|{'a':1}.get('b',0) → 0|
|.pop(key)|删除 key 并返回值|{'a':1}.pop('a') → 1|
|.update(dict2)|用另一个字典更新当前字典|{'a':1}.update({'b':2}) → {'a':1,'b':2}|
|len(dict)|获取字典键的数量|len({'a':1,'b':2}) → 2|
### 类与对象

```python
class parents:
    def __init__(self，a,b)->None:	#构造函数
        self.a = a
        self.b = b
    def func()->int:
        c = self.a + self.b
        return c
class son(parents):
    def __init__(self,a,e)->None:
        super().__init__(a)			#调用父类的构造函数
        self.e = 2		
    def func1()->None:
        super().func()	#调用父类方法 
if __name__ == "__main__":
    father = parent(a,b)
    jack = son(a,e)
    father.func()
    jack.func1()
```

###  装饰器

```python
def decorator(func):			#func指被装饰的函数
    def wrapper(*args, **kwargs):	#wrapper装饰，*arg指任意数量的位置参数, **kwargs关键字参数（字典），*用法见下词条‘解包’
    	print('write forward')
        result = func(*args, **kwargs)
        print('write behind')
        return result
    return wrapper

@decorator
def function()->None
    print("Hello, world!")

function()      
```

### 解包

在pytorch中很常见的用法

*：对非关键字的解包

```python
def func(a,b,c):
    print(a,b,c)
array = [1,2,3]
func(*array)	#使用*进行解包
```

实践用法：

```python
#比如我有一个data，因为我要数据清洗，所以我把整个train_features,train_labels,test_features,test_labels通过concat函数捏成一个data,那么当我要调用train函数时

data = torch.cat([train_features,train_labels,
                  test_features,test_labels], dim =0)
#数据清洗，略
def train(train_features,train_labels,test_features,test_labels):
    ...
#可以写成
train(*data)
```

### with打开文件，避免了打开文件忘记关闭的尴尬

```python
with open('filename', 'mode') as file:
```

moed:`'r'`: 只读模式（默认模式）。

`'w'`: 写入模式（如果文件存在，会覆盖原文件；如果文件不存在，会创建新文件）。

`'a'`: 追加模式（在文件末尾追加内容）。

```python
with open('example.txt', 'w') as file:
    file.write('Hello, World!')
```

###  语法糖

断言 assert

```python
assert i == 0
#当i==0的时候，继续函数，不然break
```

lamba，创建简单匿名函数，常用于一次性的函数，没啥卵用纯炫技

```python
lamba x, y, z: x**2+3*y-sqrt(z)
```

slcie:切片，这个在分割数据集的时候还是挺有用的

```python
slice(start, stop, step)
```

简写for循环：
```python
sum(i) for i in range(10)
```
### 杂七杂八
1.把列表或元组放到一起形成一个新的迭代器,用于遍历多个可迭代对象
```pythopn
zip(X, Y)
for x, y in zip(X, K):
```
2.

