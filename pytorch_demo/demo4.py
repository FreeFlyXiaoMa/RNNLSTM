# -*- coding: utf-8 -*-
#@Time    :2019/6/27 21:27
#@Author  :XiaoMa

import torch as t
import numpy as np
# a=t.Tensor(2,3)
# print(a)
# b=t.Tensor([[1,2,3],[4,5,6]])
# print('b\n',b.tolist())
# b_size=b.size()
# print('b_size:',b_size)
a=t.randn(3,4)
print(a)
print(a[0])
print(a[:,0])


print(a>1)
print(a[a>1])
print(a[t.LongTensor([0,1])])












