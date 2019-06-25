# -*- coding: utf-8 -*-
#@Time    :2019/6/25 22:23
#@Author  :XiaoMa

from __future__ import print_function
import torch as t
x=t.FloatTensor(5,3)
print(x)

#均匀分布
x=t.rand(5,3)
print(x)
print(x.size()[1])

y=t.rand(5,3)
print(t.add(x,y))

result=t.FloatTensor(5,3)
t.add(x,y,out=result)
print('result:\n',result)

y.add(x)
print('第一种加法：',y)
y.add_(x)
print('第二种加法：',y)