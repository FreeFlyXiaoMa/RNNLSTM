# -*- coding: utf-8 -*-
#@Time    :2019/7/4 22:15
#@Author  :XiaoMa
import torch as t
import torch.nn as nn
from torch.autograd import Variable as V

input=V(t.randn(2,3))
model=nn.Linear(3,4)
output1=model(input)    #得到输出方式1

output2=nn.functional.linear(input,model.weight,model.bias) #得到输出方式2

print(output1==output2)

b=nn.functional.relu(input)
b2=nn.ReLU(input)

#print(b==b2)




