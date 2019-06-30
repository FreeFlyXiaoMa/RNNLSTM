# -*- coding: utf-8 -*-
#@Time    :2019/6/30 19:22
#@Author  :XiaoMa

import torch as t
from torch.autograd import Variable as V
from matplotlib import pyplot as plt
from IPython import display

#为了在不同的计算机上运行时下面的输出一致，设置随机数种子
t.manual_seed(1000)
#t.set_default_tensor_type('torch.FloatTensor')

def get_fake_data(batch_size=8):
    """产生随机数据，添加了一些噪声"""
    x=t.rand(batch_size,1)*20.0
    y=x*2+(1+t.randn(batch_size,1))*3
    return x,y

#查看下x-y分布
# x,y=get_fake_data()
# plt.scatter(x,y)
# plt.show()
#随机初始化参数
w=V(t.rand(1,1),requires_grad=True)
b=V(t.zeros(1,1),requires_grad=True)
lr=0.001 #学习率

for ii in range(8000):
    x,y=get_fake_data()
    x,y=V(x),V(y)

    #forward:计算loss
    w=w.float()
    b=b.float()
    y_pred=x.mm(w)+b.expand_as(y)
    loss=0.5*(y_pred-y)**2
    loss=loss.sum()

    #自动计算梯度
    loss=loss.float()
    loss.backward()

    #更新参数
    w.data.sub_(lr*w.grad.data)
    b.data.sub_(lr*b.grad.data)

    #梯度清零
    w.grad.data.zero_()
    b.grad.data.zero_()

    if ii%1000==0:
        #画图
        display.clear_output(wait=True)
        x=t.arange(0,20).view(-1,1)
        #x=x.long()
        w.data=w.data.long()
        b.data=b.data.long()
        y=x.mm(w.data)+b.data.expand_as(x)
        plt.plot(x.numpy(),y.numpy())   #predicted

        x2,y2=get_fake_data(batch_size=20)  #得到20组样本
        plt.scatter(x2.numpy(),y2.numpy())  #真实数据

        plt.xlim(0,20)
        plt.ylim(0,41)
        plt.show()
        plt.pause(0.5)
print(w.data.squeeze()[0],b.data.squeeze()[0])