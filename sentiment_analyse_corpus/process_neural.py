# -*- coding: utf-8 -*-
#@Time    :2019/9/18 10:05
#@Author  :XiaoMa
import os

# dir='./neural'
# f=open('neural.txt','w')
# for root,dirs,files in os.walk(dir):
#     for file in files:
#         f.write(file+'\n')
# f.close()
#
# with open('satisfied.txt','w') as f:
#     for root,dir,files in os.walk('./satisfied'):
#         for file in files:
#             f.write(file+'\n')
#
# with open('unsatisfied.txt','w') as f:
#     for root,dir,files in os.walk('./unsatisfy'):
#         for file in files:
#             f.write(file+'\n')
import json

#中性态度语料获取
with open('./data/neural.txt','r',encoding='utf-8') as f:
    lines=f.readlines()
    total=''
    for line in lines:
        with open('./neural/'+line.strip(),'r',encoding='utf-8') as file:
            data=file.read()
            total=total+'\n'+data
    data_list=total.split('\n')
    data_list_=[]
    for i in range(len(data_list)):
        if data_list[i] != '':
            data_list_.append(data_list[i])

    sample=[]
    for i in range(len(data_list_)):
        item=data_list_[i]
        item=item.replace('[','').replace(']','')
        item_i=item.split('}')
        for j in item_i:
            item_j=j.split('{')
            for jj in item_j:
                if jj !='' and jj != ',':
                    sample.append(jj)

role_list=[]
content_list=[]
for item in sample:
    item_list=item.split(',')
    role=item_list[3]
    role_=role.split(':')[1]

    content=item_list[4]
    content_=content.split(':')[1]

    role_list.append(role_.replace('"',''))
    content_list.append(content_.replace('"',''))

import pandas as pd
df_neural=pd.DataFrame()
df_neural['role']=role_list
df_neural['content']=content_list
label_list=[]
for i in range(len(role_list)):
    label_list.append('0')
df_neural['label']=label_list
df_neural.to_csv(open('./data/df_neural.csv','w',encoding='utf-8'))
# =========================================================================================================

