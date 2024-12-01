### 卷积神经网络

#### 单通道卷积

```python
def conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] -h + 1, X.shape[1] - w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i : i+h, j : j+w ]*K).sum()
    return Y

class Conv(nn.Module):
    def __init__(self, kernel_size)->None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self,X)->torch.tensor:
        X = conv(X, self.weight) + self.bias
        return X


X = torch.ones((6,8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = conv(X,K)
print(Y)

conv2d = nn.Conv2d(1 , 1, kernel_size=(1,2), bias = False)
#输入的张量是四维的(batch_size , in_channels, height, width)
X = X.reshape(1,1,6,8)          #变形为单样本，单通道
Y = Y.reshape(1,1,6,7)

batch = 20
lr = 0.03
def optimize(batch,learning_rate):
    for i in range(batch):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        #更新权重
        with  torch.no_grad():
            conv2d.weight[:] -= learning_rate * conv2d.weight.grad
        if (i+1) % 2 == 0:
            print(f'batch {i+1}, loss {l.sum():.3f}')

optimize(batch,lr)

print(conv2d.weight.data.reshape(1,2))
```

#### 多通道卷积

```python
def corr2d_multi_in(X, K) :
    a = 0
    for x, k in zip(X, K):
        a += d2l.corr2d(x, k)
    return a

def corr2d_multi_in_out(X, K):
    result = []
    for k in K:
        result = corr2d_multi_in(X, K)
        result.apend(result)
    stacked_result = torch.stack(result, dim = 0)
    return stacked_result
```

#### 池化层

作用：1.让位置不那么敏感。2.对矩阵边缘裁剪



1.最大池化。进行模糊化，允许误差

 2.平均池化

> 默认步幅和池化窗口大小相同

```python
def pool(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0]- p_h +1, X.shape[1] - p_w +1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h , j:j+p_w].max()
            if mode == 'avg':
                Y[i, j] = X[i:i+p_h , j:j+p_w].mean() 
    return Y

#填充、步幅
pool = nn.MaxPool2d(3)  #3*3的最大池化层
pool2 = nn.MaxPool2d(2, padding=1, stride = (2,3))
```

#### LeNet

针对手写数据集

```python
class LeNet(nn.Module):
    def __init__(self)->None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(16*5*5, 120, bias=True)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84,10)

    def forward(self, X:torch.Tensor)->torch.Tensor:
        X = X.view(-1,1,28,28)
        X = self.conv1(X); X = self.sigmoid(X)
        X = self.avg_pool(X)
        X = self.conv2(X); X = self.sigmoid(X)
        X = self.avg_pool(X); X = self.flatten(X)
        X = self.linear1(X); X = self.sigmoid(X)
        X = self.linear2(X)
        X = self.linear3(X)
        return X
```

#### AlexNet

ImageNet数据集

从SVM的人工特征提取到CNN学习特征

```python
class Net(nn.Module):
    def __init__(self)->None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.drop_out = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dense1 = nn.Linear(6400, 4096)
        self.dense2 = nn.Linear(4096, 4096)
        self.dense3 = nn.Linear(4096, 10)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = self.relu(X)
        X = self.max_pool(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.max_pool(X)
        X = self.conv3(X)
        X = self.relu(X)
        X = self.conv4(X)
        X = self.relu(X)
        X = self.conv4(X)
        X = self.relu(X)
        X = self.max_pool(X)
        X = self.flatten(X)
        X = self.relu(X)
        X = self.drop_out(X)
        X = self.dense2(X)
        X = self.relu(X)
        X = self.drop_out(X)
        X = self.dense3(X)
        return X
```

#### VGG

> 模型小但深效果更好

多个VGG块后面接全连接层 ——>VGG架构

```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    conv_blocks.extend([
        nn.Flatten(), 
        nn.Linear(out_channels * 7 * 7, 4096), 
        nn.ReLU(), 
        nn.Dropout(p=0.5), 
        nn.Linear(4096, 4096), 
        nn.ReLU(), 
        nn.Dropout(p=0.5), 
        nn.Linear(4096, 10)
    ])
    return nn.Sequential(*conv_blocks)
```

