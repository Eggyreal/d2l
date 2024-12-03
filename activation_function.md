```python
def Sigmoid(x:float)->float:
    y = 1 / (1 + np.exp(-x))
    return y

def Tanh(x:float)->float:
    a = np.exp(x) - np.exp(-x)
    b = np.exp(x) + np.exp(-x)
    return a / b

def ReLU(x:float)->float:
    return max(0,x)

def Leaky_ReLU(x:float, a = 0.01)->float:
    if x >= 0 :
        return x
    if x < 0 :
        return a*x

def ELU(x:float, a = 0.01)->float:
    if x >= 0 :
        return x
    if x < 0 :
        return a * (np.exp(x) - 1)

def swish(x:float)->float:
    return x * Sigmoid(x)

def Softmax(X:torch.Tensor)->torch.Tensor:
    exp_X = torch.exp(X - torch.max(X))
    return exp_X / exp_X.sum()

def plot_activation(name: str, x: np.ndarray):
    # 激活函数映射
    activation_functions = {
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": lambda x: np.maximum(0, x),
        "leaky_relu": lambda x: np.where(x >= 0, x, 0.01 * x),
        "elu": lambda x: np.where(x >= 0, x, 0.01 * (np.exp(x) - 1)),
        "swish": lambda x: x * Sigmoid(x),
    }
    
    if name.lower() not in activation_functions:
        print(f"Unknown activation function: {name}")
        return

    # 计算激活值
    func = activation_functions[name.lower()]
    y = np.vectorize(func)(x)  # 将标量函数映射为数组操作
    
    # 绘制曲线
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=name, color="blue")
    plt.title(f"Activation Function: {name.capitalize()}")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# 测试绘制
x = np.linspace(-10, 10, 500)
plot_activation("sigmoid", x)
plot_activation("tanh", x)
plot_activation("relu", x)
plot_activation("leaky_relu", x)
plot_activation("elu", x)
plot_activation("swish", x)
```

