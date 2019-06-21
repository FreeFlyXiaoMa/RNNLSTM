# -*- coding: utf-8 -*-
#@Time    :2019/6/21 11:41
#@Author  :XiaoMa
#@email   :

import re
"""
global和nonlocal关键字
"""

count=0
def keyword_global():
    global count
    count+=1
    print('局部count变量=',count)

keyword_global()
print('全局count=',count)


def keyword_nonlocal():
    count2=0
    def part_bias():
        global count
        nonlocal count2
        count2+=1
        print('其他作用域count2=',count2)

    print('局部变量count2=',count2)
    return part_bias()

keyword_nonlocal()





