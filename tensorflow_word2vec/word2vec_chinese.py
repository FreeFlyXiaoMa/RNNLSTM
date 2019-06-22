# -*- coding: utf-8 -*-
#@Time    :2019/6/21 16:12
#@Author  :XiaoMa
#@email   :
from __future__ import absolute_import
from __future__ import division
from __future__ import  print_function

import collections
import math
import random
import jieba
import numpy as np
from six.moves import xrange
import tensorflow as tf

def read_data():
    """
    文本处理
    :return:
    """
    stop_words=[]
    with open('stop_words.txt','r',encoding='utf-8') as f:
        line=f.readline()
        while line:
            stop_words.append(line[:-1])
            line=f.readline()
    stop_words=set(stop_words)  #
    print('停用词读取完毕,共{n}个词'.format(n=len(stop_words)))

    #读取文本
    raw_word_list=[]
    with open('doupochangqiong.txt','r',encoding='utf-8') as f:
        line=f.readline()
        while line:
            while '\n' in line:
                line=line.replace('\n','')
            while ' ' in line:
                line=line.replace(' ','')
            if len(line) >0:
                raw_words=list(jieba.cut(line,cut_all=False))
                raw_word_list.append(raw_words)
            line=f.readline()
    return raw_word_list

#step1
words=read_data()
print('data size',len(words))

#step 建立词典，替换掉生僻词
vocabulary_size=50000

def build_dataset(words):
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    print('count',len(count))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=list()
    unk_count=0
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            index=0
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary
data,count,dictionary,reverse_dictionary=build_dataset(words)

#删除words节省内存
del words
print('most common words (+UNK)',count[:5])
print('sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])

data_index=0



