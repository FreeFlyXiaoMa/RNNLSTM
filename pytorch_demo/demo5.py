# -*- coding: utf-8 -*-
#@Time    :2019/6/27 23:25
#@Author  :XiaoMa

import torch as t
import numpy as np
# x=t.arange(27).view(3,3,3)
# print(x)
# print(x[[0,2],...])
# b=t.linspace(15,0,6).view(2,3)
# print('b=\n',b)
# c=b.t()
# print('c=\n',c)
# print(c.is_contiguous())
# d=b.contiguous()
# print('d=\n',d.is_contiguous())

a=np.ones([2,3],dtype=np.float32)
b=t.from_numpy(a)
print('b',b)
print(t.Tensor(a))
a[0,1]=100
print(a)
print('b',b)
c=b.numpy()
print('c',c)

# a=t.ones(3,2)
# b=t.zeros(2,3,1)
# c=a+b
# print('ccc=',c)
#
# a=a.view(1,3,2).expand(2,3,2)
# print('aaaa',a)
# b=b.expand(2,3,2)
# print('bbbbb',b)

a=t.arange(0,6)
print(a.storage())
b=a.view(2,3)
print(b.storage())

print(id(a.storage())==id(a.storage()))

a[1]=100
print(a)
print(b)
b[1]=200
print(b)
print(a)
#print(b[1]=200)
