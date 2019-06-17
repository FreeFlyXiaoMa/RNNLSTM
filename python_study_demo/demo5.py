"""
pickle存储数据
"""
import pickle
d={}
integers=range(9999)
d['i']=integers
f=open('22902.dat','wb')
pickle.dump(d,f)
f.close()
f=open('22903.dat','wb')
pickle.dump(d,f,True)
f.close()

import os
s1=os.stat('22902.dat').st_size #得到两个文件的大小
s2=os.stat('22903.dat').st_size
print('%d,%d,%.2f%%'%(s1,s2,(s2+0.0)/s1*100))
integers=pickle.load(open('22903.dat','rb'))
print(integers)

class Book(object):
    def __init__(self,name):
        self.name=name
    def my_book(self):
        print('my book is:',self.name)
pybook=Book('《from beginner to master》')
pybook.my_book()

from pandas import Series,DataFrame
import pandas as pd

s2=Series(data=[100,'python','showsss','qddddddd'],
       index=['mark','title','university','name'])
s2['name']='ssssssss'
ilst=['java','perl']
s5=Series(s2,index=ilst)
print(s5)

"""
DataFrame
"""
data={'name':['yahoo','google','facebook'],
      'marks':[200,400,800],
      'price':[9,3,7]
            }
f1=DataFrame(data=data)
print(f1)
f2=DataFrame(data,columns=['name','price','marks','3333'])
print(f2)

"""
字典套字典
"""
newdata={'lang':{'firstline':'python','secondline':'java'},
         'price':{'firstline':8000,}
         }
f4=DataFrame(newdata)
print(f4)

f5=DataFrame(newdata,index=['firstline','seconline','thirdline'])
#print(f5)
print(f5['lang'])


my_generator=(x*x for x in range(4))
print(my_generator)

my_list=[x*x for x in range(4)]
print(my_list)

print(dir(my_generator))

for i in my_generator:
    print(i)
for i in my_list:
    print(i)

import numpy
import math
re=2*(pow(16,6))+(15*(pow(16,5))+15*(pow(16,4))+15*(pow(16,3))+15*(pow(16,2))+15*(pow(16,1))+15)*2
print('最终结果：',re)
s1=re/1024  #K
s2=s1/1024  #M
print(s2)


# 二进制的转换
def Dec2Bin(dec):
    result = ''

    if dec:
        result = Dec2Bin(dec // 2)
        return result + str(dec % 2)
    else:
        return result

bit=Dec2Bin(re)
print(bit)


