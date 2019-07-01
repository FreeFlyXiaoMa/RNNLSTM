# -*- coding: utf-8 -*-
#@Time    :2019/7/1 22:41
#@Author  :XiaoMa
import torch as t
from torch.autograd import Variable as V
from torch import nn
t.manual_seed(1000)#设置随机数种子，保证每次运行得到相同的结果
input=V(t.randn(2,3,4)) #batch_size=3,序列长度为2，序列中每个元素占4维

#LSTM输入向量为4维，3个隐藏元，1层
lstm=nn.LSTM(4,3,1)

#初始状态：1层，batch_size=3,3个隐藏元
h0=V(t.randn(1,3,3))
c0=V(t.randn(1,3,3))

out,hn=lstm(input,(h0,c0))
print(out)


















