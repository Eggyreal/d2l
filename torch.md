## torch 

> 深度学习最重要的
>
> 数据结构：tensor，不是不属于*args和**kwargs，不能解包

```python
import torch
```

创建tensor

```python
tensor1 = torch.tensor([1,2,3,4], dtype=torch.float)
tensor = torch.FloatTensor([1,2,3,4])
tensor2 = torch.arange(12)
tensor_reshape = tensor.reshape(1,3,4)  #3 dimentions
tensor1 = tensor1.reshape(-1, 2)  #-1可以做占位符
x = torch.randn(12)  #正态分布的随机数
x = torch.rand(12)  #均匀分布的随机数
x = torch.randn(size=(2,20))
x = torch.randn(2,20)
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
#### 参数管理
```python
net[2].state_dict()     #获得net[2]的权重和偏置参数
net[2].bias             #获得net[2]的偏置
print(net[2].bias.data) #把偏置变成一个tensor
print(*[(name,param.shape) for name, param in net[0].named_parameters()])
net.named_parameters()   #获取所有参数，保存在生成器（一个形式为（name，parameters）的元组）
```
### 常见torch.nn.init中的函数,网络初始化。
具体代码见[net.py](net.py)
```python
nn.init.normal_(m.weight, mean=0, std=0.01)  #正态分布来初始化权重
nn.init.zeros_(m.bias)                       #偏置初始化为零
nn.init.constant_(m.weight, 1)               #初始化为常数
nn.init.xavier_uniform_                      #xavier 初始化，常见于Sigmoid 或 Tanh激活函数
# 函数后面带“_”表示原地操作

#均匀分布初始化
torch.nn.init.uniform_(tensor, a=0.0, b=1.0) #tensor: 需要初始化的张量。a: 均匀分布的下界（默认为 0.0）。b: 均匀分布的上界（默认为 1.0）。
```
参数绑定
```python
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8), nn.ReLu(), shared, nn.ReLU(), shared)
#此时 net[2],net[4]的权重在是一样的
```
自定义层, 以全连接层做例子
```python
class MyLinear(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size,input_size))
        self.bias = nn.Parameter(torch.randn(output_size))
    def forward(self,X):
        X = torch.matmul(X,self.weight.T) + self.bias
        return X
```
#### 读写文件
```python
x = torch.arange(10)
y = torch.zeros(4)
my_dict = {'x':x,
           'y':y}
torch.save(my_dict,'mydict')       #以pickle格式保存文件，后缀为.pt/.pth
my_dict2 = torch.load('mydict')    #读取文件
```
```python
class MLP(nn.Module):
    ...
net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)
torch.save(net.state_dict(),'mlp.params') #保存所有的parameters

clone = MLP()   #创建clone
clone.load_state_dict(torch.load('mlp.params'))     #clone.load_state_dict,把字典加载到模型中
clone.eval()            #评估模式，不求梯度
```
### 卷积CNN convolution
> 一种特殊的全连接层 >
提取局部特征
二维交叉相关
具体代码看[convolution.py](https://github.com/Eggyreal/d2l/blob/main/convolution.py)

1. 平移不变性
   比如说一张图片的识别，一只狗放在屏幕左上角和右下角都应该是狗，虽然他们的数据肯定不一样，但是都要识别出来是狗。
2. 局部性
```python
conv2d = nn.conv2d = nn.Conv2d(1 , 1, kernel_size=(1,2), bias = False)  #输入通道，输出通道，卷积核形状，偏置
#输入的张量是四维的(batch_size , in_channels, height, width)
X = X.reshape(1,1,6,8)          #变形为单样本，单通道
Y = Y.reshape(1,1,6,7)
```
