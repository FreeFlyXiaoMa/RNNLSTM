# -*- coding: utf-8 -*-
#@Time    :2019/6/29 18:36
#@Author  :XiaoMa
import torch as t
from torch.autograd import Variable as V
def f(x):   #计算输出值
    y=x**2*t.exp(x)
    return y

def gradf(x):   #手动计算梯度
    dx=2*x*t.exp(x)+x**2*t.exp(x)

    return dx

x=V(t.randn(3,4),requires_grad=True)
y=f(x)
print('y==',y)

y.backward(t.ones(y.size()))    #backward()反向传播时，自动计算梯度
print('x.grad=',x.grad)
gr=gradf(x)
print('手动计算梯度:',gr)




