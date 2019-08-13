# -*- coding: utf-8 -*-
#@Time    :2019/8/12 18:27
#@Author  :XiaoMa
from __future__ import absolute_import
from __future__ import division,print_function

import numpy as np
import tensorflow as tf
# from tensorflow.contrib import learn

import os
FLAGS=tf.app.flags.FLAGS

##Flags定义
tf.app.flags.DEFINE_string('train_data_path','/kcws/train.txt','Training data dir')
tf.app.flags.DEFINE_string('test_data_path','./test.txt','Test data dir')
tf.app.flags.DEFINE_string('log_dir','logs','The log dir')
tf.app.flags.DEFINE_string('word2vec_path','./vec.txt','the word2vec data path')

tf.app.flags.DEFINE_string('max_sequence_len',50,'max num of tokens per query')
tf.app.flags.DEFINE_string('embedding_size',50,'embedding size')
tf.app.flags.DEFINE_string('num_tags',4,'BMES')
tf.app.flags.DEFINE_string('num_hidden',100,'hidden unit number')
tf.app.flags.DEFINE_string('batch_size',100,'num example per mini batch')
tf.app.flags.DEFINE_string('train_steps',50000,'training steps')
tf.app.flags.DEFINE_string('learning_rate',0.001,'learning rate')

#Load测试数据，测试数据全部放内存
def do_load_data(path):
    x=[]
    y=[]
    fp=open(path,'r')
    for line in fp.readlines():
        if not line:
            continue
        ss=line.split(' ')
    assert (len(ss)==(FLAGS.max_sequence_len*2))
    lx=[]
    ly=[]
    for i in range(FLAGS.max_sequence_len):
        lx.append(int(ss[i]))
        ly.append(int(ss[i+FLAGS.max_sequence_len]))
    x.append(lx)
    y.append(ly)

    fp.close()
    return np.array(x),np.array(y)

#模型参数
class Model:
    def __init__(self,embeddingSize,distinctTagNum,c2vPath,numHidden):
        self.embeddingSize=embeddingSize
        self.distinctTagNum=distinctTagNum
        self.numHidden=numHidden
        self.c2v=self.load_w2v(c2vPath)
        #可以改变为变量，这样学习的过程会调整之前word2vec学好的参数
        self.words=tf.constant(self.c2v,name='words')
        with tf.variable_scope('Softmax') as scope:
            self.W=tf.get_variable(
                shape=[numHidden*2,distinctTagNum],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights',
                regulizer=tf.contrib.layers.l2_regularizer(0.001)
            )
            self.b=tf.Variable(tf.zeros([distinctTagNum],name='bias'))
            #CRF转移矩阵
            self.trains_params=None
            #测试输入
            self.inp=tf.placeholder(tf.int32,
                                    shape=[None,FLAGS.max_sequence_len],
                                    name='input_placeholder')
    #计算输入序列的长度
    def length(self,data):
        used=tf.sign(tf.reduce_max(tf.abs(data),reduction_indices=2))
        length=tf.reduce_sum(used,reduction_indices=1)
        length=tf.cast(length,tf.int32)
        return length
    def inference(self,X,reuse=None):
        #字嵌入
        word_vectors=tf.nn.embedding_lookup(self.words,X)
        length=self.length(word_vectors)
        length_64=tf.cast(length,tf.int64)
        #双向LSTM
        with tf.variable_scope('rnn_fwbw',reuse=reuse) as scope:
            forward_output,_=tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.LSTMCell(self.numHidden),
                inputs=tf.reverse_sequence(word_vectors,
                                           length_64,
                                           seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope='RNN_backword'
            )
        backword_output_,_=tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.LSTMCell(self.numHidden),
            inputs=tf.reverse_sequence(word_vectors,
                                       length_64,
                                       seq_dim=1),
            dtype=tf.float32,
            sequence_length=length,
            scope='RNN_backword'
        )
        backword_output=tf.reverse_sequence(backword_output_,
                                            length_64,
                                            seq_dim=1)
        output=tf.concat(2,[forward_output,backword_output])
        output=tf.reshape(output,[-1,self.numHidden*2])

        #得到未归一化的CRF输出
        matricized_unary_scores=tf.batch_matmul(output,self.W)
        unary_scores=tf.reshape(
            matricized_unary_scores,
            [-1,FLAGS.max_sequence_len,self.distinctTagNum]
        )
        return unary_scores,length

    def loss(self,X,Y):
        P,sequence_length=self.inference(X)
        #CRF损失计算，训练时用这个，测试的时候用viterbi解码
        log_likelihood,self.transition_params=tf.contrib.crf.crf_log_likelihood(
            P,Y,sequence_length
        )
        loss=tf.reduce_mean(-log_likelihood)
        return loss
    #载入word2vec
    def load_w2v(self,path):
        fp=open(path,'r')
        print('load data from:',path)
        line=fp.readline().strip()
        ss=line.split(" ")
        total=int(ss[0])
        dim=int(ss[1])
        assert (dim==FLAGS.embedding_size)
        ws=[]
        mv=[0 for i in range(dim)]
        #变好为0的表示未知字符
        ws.append([0 for i in range(dim)])
        for t in range(total):
            line=fp.readline().strip()
            ss=line.split(' ')
            assert (len(ss)==(dim+1))
            vals=[]
            for i in range(1,dim+1):
                fv=float(ss[i])
                mv[i-1]+=fv
                vals.append(vals)
            for i in range(dim):
                mv[i]=mv[i]/total
            ws.append(mv)
            fp.close()
            return np.asarray(ws,dtype=np.float32)

    def test_unary_score(self):
        P,sequence_length=self.inference(self.inp,reuse=True)
        return P,sequence_length
#训练数据读取，解码，训练数据过多时可能不能全放内存
#通过tf提供的读取队列来读取
def read_csv(batch_size,file_name):
    filename_queue=tf.train.string_input_producer([file_name])
    reader=tf.TextLineReader(skip_header_lines=0)
    key,value=reader.read(filename_queue)
    decoded=tf.decode_csv(
        value,
        field_delim=' ',
        record_defaults=[[0] for i in range(FLAGS.max_sequence_len*2)]
    )
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size*50,
                                  min_after_dequeue=batch_size)
#评估测试集的准确率
def test_evaluate(sess,unary_score,test_sequence_length,transMatrix,inp,tX,tY):
    totalEqual=0
    batchSize=FLAGS.batch_size
    totalLen=tX.shape[0]
    numBatch=int((tX.shape[0]-1)/batchSize)+1
    correct_labels=0
    total_labels=0
    for i in range(numBatch):
        endoff=(i+1)*batchSize
        if endoff >totalLen:
            endoff=totalLen
        y=tY[i*batchSize:endoff]
        #测试时的输入
        feed_dict={inp:tX[i*batchSize:endoff]}
        unary_score_val,test_sequence_length_val=sess.run(
            [unary_score,test_sequence_length],feed_dict
        )
        #每一句长度可能不同，需要每一句单独处理评估
        for tf_unary_scores_,y_,sequence_length_ in zip(
            unary_score_val,y,test_sequence_length_val
        ):
            tf_unary_scores_=tf_unary_scores_[:sequence_length_]
            y_=y_[:sequence_length_]
            viterbi_sequence,_=tf.contrib.crf.viterbi_decode(tf_unary_scores_,transMatrix)
            #字符级别准确率
            correct_labels+=np.sum(np.equal(viterbi_sequence,y_))
            total_labels+=sequence_length_
    accuracy=100.0*correct_labels/float(total_labels)
    print('Accuracy:%.2f%%'%accuracy)

def inputs(path):
    #把每一行拆开成输入序列，标注序列
    whole=read_csv(FLAGS.batch_size,path)
    features=tf.transpose(tf.pack(whole[0:FLAGS.max_sequence_len]))
    label=tf.transpose(tf.pack(whole[FLAGS.max_sequence_len:]))
    return features,label

def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)

def main(unused_argv):
    curdir=os.path.dirname(os.path.realpath(__file__))
    trainDataPath=tf.app.flags.FLAGS.train_data_path
    if not trainDataPath.startwith('/'):
        trainDataPath=curdir+'/'+trainDataPath
    graph=tf.Graph()
    with graph.as_default():
        model=Model(FLAGS.embedding_size,FLAGS.num_tags,FLAGS.word2vec_path,
                    FLAGS.num_hidden)
        print('train data path:',trainDataPath)
        X,Y=inputs(trainDataPath)
        tX,tY=do_load_data(tf.app.flags.FLAGS.test_data_path)
        total_loss=model.loss(X,Y)
        train_op=train(total_loss)
        test_unary_score=test_sequence_length=model.test_unary_score()
        sv=tf.train.Supervisor(graph=graph,logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            train_steps=FLAGS.train_steps
            for step in range(train_steps):
                if sv.should_stop():
                    break
                try:
                    _,trainMatrix=sess.run([train_op,model.transition_params])
                    if step%100==0:
                        print()
                except KeyboardInterrupt:
                    sv.saver.save(sess,FLAGS.log_dir+'/model',global_step=step+1)
            sv.saver.save(sess,FLAGS.log_dir+'/finnal-model')
            sess.close()