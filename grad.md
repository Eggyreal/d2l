## Grad

### using torch to calculate the grad

```python
import torch 
x = torch.ones(2,2,requires_grad = True)  #crate a tensor, the requirement of grad is passed

y = x + 2
z = y * y * 3

z.backward(torch.ones_like(z))  #calculate dz/dx
#by using chain rule we will see, dz/dx=dz/dy * dy/dx
#which equals to dz/dx = 6y * 1
print(x.grad)  #[[18,18],[18,18]]

#renew the x
lr = 0.01  #learning_rate
x_new = x - lr*x.grad
```



