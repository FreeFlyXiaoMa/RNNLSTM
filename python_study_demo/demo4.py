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

import re
p=re.compile('ab')
str1='abfffa'
if p.match(str1):
    print(p.match(str1).group())

print(re.findall('a+b','abvvvvvvvvaabaaab'))
print(re.split(':','str1:str2:str3'))
print(re.split('a+b','vabmaabnaaab'))

sentence='hi jack:\n' \
         'Python is a beautiful language\n' \
         'BR'

sentences=sentence.split('\n')
print(sentences)
pattern=re.compile('(BR|language)$')
for i in sentences:
    m=re.search(pattern,i)
    if m:
        print(i)
#找出字母g后面的字母不是u
words=['gold','Google','Sougu','guess']
pattern=re.compile('.*g[^u]')
for i in words:
    m=re.search(pattern,i)
    if m:
        print(i)

pattern=re.compile(r'[1-5][0-9]')
list1=[10,20,30,40,2,3,59,60,'aa','3aaa']

match=re.findall(pattern,str(list1))

if len(match)>=0:
    print(match)

time1='10:00,99:90,8:00,19:19,14:00pm,5:xm,6,00,8:0923:23pm,' \
      '29:19pm.23:59'
pattern=r'\b([01]?[0-9]|2[0-4])(:)([0-5][0-9])'

match=re.findall(pattern,time1)
print([''.join(x) for x in match])
