import torch

X = torch.randn(X,dtype = torch.float)
#构造网络层
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
net = Net(X)
#顺序块，李沐傻逼写着炫技的，还不如像上面老老实实地写
class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for block in args:
            self._modules[block] = block
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256,10))
