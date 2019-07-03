# -*- coding: utf-8 -*-
#@Time    :2019/7/2 14:04
#@Author  :XiaoMa

import torch as t
from torch.autograd import Variable as V
from torch import nn
t.manual_seed(1000)#设置随机数种子，保证每次运行得到相同的结果
input=V(t.randn(2,3,4)) #batch_size=3,序列长度为2，序列中每个元素占4维


input=V(t.randn(2,3,4))
#一个LSTMCell对应的层数只能是一层
lstm=nn.LSTMCell(4,3)
hx=V(t.randn(3,3,))
cx=V(t.randn(3,3))
out=[]

for i_ in input:
    hx,cx=lstm(i_,(hx,cx))
    out.append(hx)
print(t.stack(out))

#pytorch中的embedding层
embedding=nn.Embedding(4,5) #有4个词，每个词有5维向量表示

#可以用预训练好的词向量初始化embedding
embedding.weight.data=t.arange(0,20).view(4,5)

input=V(t.arange(3,0,-1).long())
output=embedding(input)
print('output=',output)








