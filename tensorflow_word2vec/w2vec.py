# -*- coding: utf-8 -*-
#@Time    :2019/8/27 16:04
#@Author  :XiaoMa
import time
from gensim.models import Word2Vec
import codecs

#注意这里的corpus是所有字的集合
def word2vec(model_path,corpus,embedding_size=256,min_count=1,window=7):
    print('开始训练word2vec:%s'%time.ctime())
    model=Word2Vec(sentences=corpus,size=embedding_size,min_count=min_count,window=window,workers=4,iter=10)
    model.save(model_path)
    with codecs.open(model_path,mode='w') as file:
        for word,_ in model.wv.vocab.items():
            vector=[str(i) for i in model.wv[word]]
            file.write(word+' '+' '.join(vector)+'\n')


























