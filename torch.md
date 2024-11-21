## torch 

> 深度学习最重要的
>
> 数据结构：tensor，不是不属于*args和**kwargs，不能解包

```python
import torch
```

创建tensor

```python
tensor1 = torch.tensor([1,2,3,4])
tensor2 = torch.arange(12)
tensor_reshape = tensor.reshape(1,3,4)  #3 dimentions
tensor1 = tensor1.reshape(-1, 2)  #-1可以做占位符
x = torch.randn(12)  #随机数
```

转换

```python
tensor1.item()  #将单元素张量转化为python标量
tensor2.tolist()	#  张量变列表
```

基础运算

```python
torch.sum(axis=0)
torch.mean()
torch.randn()    #根据正态分布生成随机tensor
torch.uniform_(a, b)  #根据均匀分布生成随机tensor
nn.paramenter()  #将一个张量包裹在 nn.Parameter 中，可以让这个张量被视为模型的参数，从而在模型优化过程中自动计算其梯度。
```

张量裁剪

 ```python
 torch.clamp(input, min=None, max=None, out=None)
 clipped_preds = torch.clamp(net(features), 1, 2，float('inf'))    #float('inf')代表没有最大值
 ```

构造简单的顺序网络	

> 构造自定义网络方法，看net.py文件

```python
#简单的线性
model = nn.Sequential(
    nn.Linear(10, 20),  # 输入10维，输出20维
    nn.ReLU(),          # 激活函数
    nn.Linear(20, 1)    # 输入20维，输出1维
)
```

