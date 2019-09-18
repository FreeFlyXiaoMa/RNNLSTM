# -*- coding: utf-8 -*-
#@Time    :2019/9/18 11:28
#@Author  :XiaoMa
import pandas as pd

with open('./data/satisfied.txt','r') as f:
    sample=[]
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        with open('./satisfied/'+line,'r',encoding='utf-8') as ff:
            content=ff.read()
            content=content.replace('[','').replace(']','')
            content_i=content.split('}')
            for i in content_i:
                i=i.strip()
                i=i.split('{')
                for ii in i:
                    if ii !='' and ii != ',':
                        sample.append(ii)

role_list=[]
content_list=[]
label_list=[]
for item in sample:
    item_list=item.split(',')
    role=item_list[3]
    role_=role.split(':')[1]
    role_list.append(role_.replace('"',''))

    content=item_list[4]
    content_=content.split(':')[1]
    content_list.append(content_.replace('"',''))

for i in range(len(role_list)):
    label_list.append('1')


df=pd.DataFrame()
df['role']=role_list
df['content']=content_list
df['label']=label_list
df.to_csv(open('./data/df_satisfied.csv','w',encoding='utf-8'))
print(df.head())