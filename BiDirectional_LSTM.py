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
for t,c,n in zip(D[1],D[2],D[3]):
    start=t.find(n) #公司名的起点索引
    if start !=-1:
        train_data.append((t,c,n))

if not os.path.exists('all_chars_me.json'):
    chars={}
    for d in tqdm(iter(train_data)):    #iter--迭代函数
        for c in d[0]: #text中的每一个字
            chars[c]=chars.get(c,0)+1   #找到c所在的位置
    chars={i:j for i,j in chars.items() if j>=min_count}
    id2char={i+2:j for i,j in enumerate(chars)}
    char2id={j:i for i,j in id2char.items()}
    json.dump([id2char,char2id],open('all_chars_me.json','w'))
else:
    id2char,char2id=json.load(open('all_chars_me.json'))

if not os.path.exists('random_order_train.json'):
    random_order=list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(random_order,
              open('random_order_train.json','w'),
              indent=4)
else:
    random_order=json.load(open('random_order_train.json'))

#从训练集中按1：9的比例抽取出验证集、
dev_data=[train_data[j] for i,j in enumerate(random_order) if j%9==mode]

#读取测试集数据
D=pd.read_csv('extract_eval.csv',encoding='utf-8',header=None)
test_data=[]
for id,t,c in zip(D[0],D[1],D[2]):
    test_data.append((id,t,c))

#位置补齐方法
def seq_padding(X,padding=0):
    L=[len(x) for x in X]
    ML=max(L)
    return np.array([
        np.concatenate([x,[padding]*(ML-len(x))]) if len(x)<ML else x for x in X
    ])

#
class data_generator:
    def __init__(self,data,batch_size=64):
        self.data=data
        self.batch_size=batch_size
        self.steps=len(self.data)//self.batch_size
        if len(self.data) % self.batch_size !=0:
            self.steps+=1
    def __le__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs=list(range(len(self.data)))
            np.random.shuffle(idxs) #顺序打乱，防止每次产生的数据相同
            X,C,S1,S2 =[],[],[],[]
            for i in idxs:
                d=self.data[i]
                text=d[0]   #信息主体
                x=[char2id.get(c,1) for c in text]
                c=class2id[d[1]] #类别对应的id
                s1,s2=np.zeros(len(text)),np.zeros(len(text))
                start=text.find(d[2])   #公司信息起点
                end=start+len(d[2])-1   #公司信息终点索引
                s1[start]=1
                s2[end]=1
                X.append(x)
                C.append(c)
                S1.append(s1)
                S2.append(s2)
                if len(X) == self.batch_size or i == idxs[-1]:
                    X=seq_padding(X)
                    C=seq_padding(C)
                    S1=seq_padding(S1)
                    S2=seq_padding(S2)
                    yield [X,C,S1,S2],None
                    X,C,S1,S2=[],[],[],[]

from keras.layers import *
from keras.models import Model
import keras.backend as  K
from keras.callbacks import Callback
from keras.optimizers import Adam

#多头注意力机制
#class Attention(Layer):
x_in=Input(shape=(None,)) #待识别句子输入
c_in=Input(shape=(1,))
s1_in=Input(shape=(None,)) #实体左边界（标签）
s2_in=Input(shape=(None,))  #实体右边界（标签）

x,c,s1,s2=x_in,c_in,s1_in,s2_in
x_mask=Lambda(lambda x:K.cast(K.greater(K.expand_dims(x,2),0),'float32'))(x)

x=Embedding(len(id2char)+2,char_size)(x)



