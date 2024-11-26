## python 基础用法

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
