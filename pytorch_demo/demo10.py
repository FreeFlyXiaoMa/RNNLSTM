# -*- coding: utf-8 -*-
#@Time    :2019/6/30 21:07
#@Author  :XiaoMa
import torch as t
from torch import nn
from torch.autograd import Variable as V


#全连接网络
class Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super(Linear,self).__init__()
        self.w=nn.Parameter(t.randn(in_features,out_features))
        self.b=nn.Parameter(t.randn(out_features))

    def forward(self, x):
        tt=t.mm(x,self.w)
        #tt=x.mm(self.w)
        return tt+self.b.expand_as(tt)

layer=Linear(4,3)
input=V(t.randn(2,4))
output=layer(input)

print(output)

for name,parameter in layer.named_parameters():
    print(name,parameter)

#带有隐层的网络
class Perception(nn.Module):
    def __init__(self,in_features,hidden_features,out_features):
        nn.Module.__init__(self)    #super(Perception,self).__init__()
        self.layer1=Linear(in_features,hidden_features) #利用之前的Linear子Module
        self.layer2=Linear(hidden_features,out_features)

    def forward(self, x):
        x=self.layer1(x)
        x=t.sigmoid(x)
        return self.layer2(x)

perception=Perception(3,4,1)
for name,param in perception.named_parameters():
    print(name,param)

