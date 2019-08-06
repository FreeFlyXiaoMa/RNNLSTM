# -*- coding: utf-8 -*-
#@Time    :2019/8/6 14:40
#@Author  :XiaoMa

a=10
def global_test():
    global a
    a=2018
    print('global_test:a=%d'%(a))
global_test()
print('a==',a)