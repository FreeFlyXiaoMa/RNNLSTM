# -*- coding: utf-8 -*-
#@Time    :2019/6/26 21:01
#@Author  :XiaoMa

import torch as t
import numpy as np

from torch.autograd import Variable
x=Variable(t.ones(2,2),requires_grad=True)
print(x)




