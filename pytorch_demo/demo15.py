# -*- coding: utf-8 -*-
#@Time    :2019/7/3 22:31
#@Author  :XiaoMa

from torch import nn as nn
import torch as t
from torch.autograd import Variable as V
#定义一个LeNet网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,3)
        )

        self.classifier=nn.Sequential(\
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
            )
    def forward(self, x):
        x=self.features(x)
        x=x.view(-1,16*5*5)
        x=self.classifier(x)
        return x
net=Net()

from torch import optim #优化器
optimizer=optim.SGD(params=net.parameters(),lr=1)
optimizer.zero_grad()   #梯度清零，相当于net.zero_grad()

input=V(t.randn(1,3,32,32))
output=net(input)
output.backward(output)     #fake backward
optimizer.step()    #执行优化

#为不同子网络设置不同的学习率，在finetune中经常用到
#如果对某个参数不指定学习率，就使用默认学习率
optimizer=optim.SGD(
    [{'param':net.features.parameters()},    #学习率为1e-5
    {'param':net.classifier.parameters(),'lr':1e-2}],lr=1e-5
)

#只为两个全连接层设置较大的学习率，其余层的学习率较小
special_layers=nn.ModuleList([net.classifier[0],net.classifier[3]])
special_layers_params=list(map(id,special_layers.parameters()))
base_params=filter(lambda p:id(p) not in special_layers_params,net.parameters())

optimizer=t.optim.SGD([
    {'param':base_params},
    {'param':special_layers.parameters(),'lr':0.01}
],lr=0.001)


