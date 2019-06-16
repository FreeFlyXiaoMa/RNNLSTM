'''
datetime库
'''
from datetime import datetime,date,time
print('当前时间：',datetime.now())

today=datetime.now()

print('日期：',today.date())
print('时间：',today.time())

print('ctime：',today.ctime())
print('tctnow:',today.utcnow())
date1=date(2019,6,16)
time1=time(20,36,40)

print('合成时间：',datetime.combine(date1,time1))

"""
math模块
"""
import math
print('取整，舍去小数部分：',math.trunc(2.3))
print('大于或等于x的整数：',math.ceil(3.4))
print('四舍五入：',round(3.5))

import re
m=re.search('(?<=abc)def','abcdef')
print(m.group(0))

m=re.search('(?<=-)\w+','spam-egg') #匹配从-开始的任意数字、字母及下划线
print(m.group(0))











