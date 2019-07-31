# -*- coding: utf-8 -*-
#@Time    :2019/7/10 18:30
#@Author  :XiaoMa

def keyword_nonlocal():
    count2=0
    def part_bias():
        nonlocal count2
        # count2+=1
        count2=count2+1
        print('其他作用域count2=',count2)


    print('局部变量count2=',count2)
    return part_bias()
keyword_nonlocal()