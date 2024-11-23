from torch import nn
import torch  


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