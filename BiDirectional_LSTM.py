import json
import tqdm
import os
import numpy as np
import pandas as pd

mode=0
min_count=2
char_size=128
maxlen=256

#读取数据，排除“其他”类型
D=pd.read_csv('extract_train.csv',encoding='utf-8',header=None)
D=D[D[2]!=u'其他']
D=[D[1].str.len() < maxlen] #最长256个字

if not os.path.exists('classes.json'):
    id2class=dict(enumerate(D[2].unique()))#共有多少重类别
    class2id={j:i for i,j in id2class.items()}
    json.dump([id2class,class2id],open('classes.json','w'))
else:
    id2class,class2id=json.load('classes.json')

#训练数据，D[1]--主体信息  D[2]--类别信息  D[3]--公司名
train_data=[]












