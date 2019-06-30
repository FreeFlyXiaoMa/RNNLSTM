# -*- coding: utf-8 -*-
#@Time    :2019/6/30 18:19
#@Author  :XiaoMa
import torch as t
from torch.autograd import Function,Variable

class MyltiplyAdd(Function):
    @staticmethod
    def forward(ctx, w,x,b):
        print('type in forward',type(x))
        ctx.save_for_backward(w,x)
        out=w*x+b
        return out

    @staticmethod
    def backward(ctx, grad_output):
        w,x=ctx.saved_variables
        print('type in backward',type(x))
        grad_w=grad_output*w
        grad_b=grad_output*1
        grad_x=grad_output*x

        return grad_w,grad_x,grad_b

x=Variable(t.ones(1))
w=Variable(t.rand(1),requires_grad=True)
b=Variable(t.rand(1),requires_grad=True)
print('开始前向传播：')
z=MyltiplyAdd.apply(w,x,b)
print('开始反向传播：')
# z.backward()
# print(x.grad,w.grad,b.grad)
print(z.grad_fn.apply(Variable(t.ones(1))))







