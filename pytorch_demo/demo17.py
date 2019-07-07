# -*- coding: utf-8 -*-
#@Time    :2019/7/7 12:11
#@Author  :XiaoMa
import torch as t
from torch import nn as nn
"""
词向量
"""
#Embedding(num_embeddings,embedding_dim)
embedding=nn.Embedding(10,2)    #10个词，每个词2维
input=t.arange(0,6).view(3,2).long()    #三个句子，每个句子有两个词

input=t.autograd.Variable(input)
output=embedding(input)
print(output.size())
print(embedding.weight.size())



