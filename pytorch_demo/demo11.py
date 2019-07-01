# -*- coding: utf-8 -*-
#@Time    :2019/7/1 13:34
#@Author  :XiaoMa

import torch as t
from torch import nn
#Sequential的三种写法
net1=nn.Sequential()
net1.add_module('conv',nn.Conv2d(3,3,3))
net1.add_module('batchnorm',nn.BatchNorm2d(3))
net1.add_module('activation_layer',nn.ReLU())

net2=nn.Sequential(nn.Conv2d(3,3,3),
                   nn.BatchNorm2d(3),
                   nn.ReLU()
                   )

from collections import OrderedDict
net3=nn.Sequential(OrderedDict([
    ('conv1',nn.Conv2d(3,3,3)),
    ('bh1',nn.BatchNorm2d(3)),
    ('al',nn.ReLU())
]))

print('net1',net1)
print('net2',net2)
print('net3',net3)

#可根据名字或序号取出子module
print(net1.conv,net2[0],net3.conv1)


