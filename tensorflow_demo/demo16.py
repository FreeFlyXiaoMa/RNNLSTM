# -*- coding: utf-8 -*-
#@Time    :2019/8/18 21:07
#@Author  :XiaoMa
import struct
import gzip
import os
from six.moves.urllib.request import urlretrieve
import numpy as np
import tensorflow as tf

def maybe_download(url, filename, expected_bytes, force=False):
  """如果不存在，请下载文件，并确保其大小合适."""
  if force or not os.path.exists(filename):
    print('试图下载文件:', filename)
    filename, _ = urlretrieve(url + filename, filename)
    print('\n已完成文件下载!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('找到并验证', filename)
  else:
    raise Exception(
      '无法验证' + filename + '. 你能用浏览器找到它吗?')
  return filename


def read_mnist(fname_img, fname_lbl):
    print('\n读取文件 %s 和 %s' % (fname_img, fname_lbl))

    with gzip.open(fname_img) as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        print(num, rows, cols)
        img = (np.frombuffer(fimg.read(num * rows * cols), dtype=np.uint8).reshape(num, rows * cols)).astype(np.float32)
        print('(Images) 返回张量的形状 ', img.shape)

        img = (img - np.mean(img)) / np.std(img)

    with gzip.open(fname_lbl) as flbl:
        # flbl.read(8) 读取最多8个字节
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.frombuffer(flbl.read(num), dtype=np.int8)
        print('（标签）返回一个张量的形状: %s' % lbl.shape)
        print('样例标签: ', lbl[:10])

    return img, lbl

# 如果需要，下载数据
url = 'http://yann.lecun.com/exdb/mnist/'
# training data
maybe_download(url,'train-images-idx3-ubyte.gz',9912422)
maybe_download(url,'train-labels-idx1-ubyte.gz',28881)
# testing data
maybe_download(url,'t10k-images-idx3-ubyte.gz',1648877)
maybe_download(url,'t10k-labels-idx1-ubyte.gz',4542)

# 读取训练数据和测试数据
train_inputs, train_labels = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test_inputs, test_labels = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

WEIGHTS_STRING = 'weights'
BIAS_STRING = 'bias'

batch_size = 100

img_width, img_height = 28,28
input_size = img_height * img_width
num_labels = 10

tf.reset_default_graph()

tf_inputs=tf.placeholder(dtype=tf.float32,shape=[batch_size,input_size],name='inputs')
tf_labels=tf.placeholder(dtype=tf.float32,shape=[batch_size,input_size],name='labels')

#定义tensorflow变量
def define_net_parameters():
    with tf.variable_scope('layer1'):
        tf.get_variable(name=WEIGHTS_STRING,shape=[input_size,500],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.02))
        tf.get_variable(name=BIAS_STRING,shape=[500],initializer=tf.random_normal_initializer(0,0.01))
    with tf.variable_scope('layer2'):
        tf.get_variable(name=WEIGHTS_STRING,shape=[500,250],initializer=tf.random_normal_initializer(0,0.02))
        tf.get_variable(name=BIAS_STRING,shape=[250],initializer=tf.random_normal_initializer(0,0.01))
    with tf.variable_scope('output'):
        tf.get_variable(WEIGHTS_STRING,shape=[250,10],initializer=tf.random_normal_initializer(0,0.02))
        tf.get_variable(BIAS_STRING,shape=[10],initializer=tf.random_normal_initializer(0,0.01))

def inference(x):
    with tf.variable_scope('layer1',reuse=True):
        w,b=tf.get_variable(WEIGHTS_STRING),tf.get_variable(BIAS_STRING)
        tf_h1=tf.nn.relu(tf.matmul(x,w)+b,name='hidden1')
    with tf.variable_scope('layer2',reuse=True):
        w,b=tf.get_variable(WEIGHTS_STRING),tf.get_variable(BIAS_STRING)
        tf_h2=tf.nn.relu(tf.matmul(tf_h1,w)+b,name='hidden2')

    with tf.variable_scope('output',reuse=True):
        w,b=tf.get_variable(WEIGHTS_STRING),tf.get_variable(BIAS_STRING)
        tf_logits=tf.nn.bias_add(tf.matmul(tf_h2,w),b)
    return tf_logits

define_net_parameters()

#定义损失
tf_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_labels,
                                                                  logits=inference(tf_inputs)))

#定义优化函数
tf_loss_minimize=tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(tf_loss)
tf_predictions=tf.nn.softmax(inference(tf_inputs))

session=tf.InteractiveSession()
tf.global_variables_initializer().run()
NUM_EPOCHS=50

def accuracy(prediction,labels):
    """预测标签准确度"""
    return np.sum(np.argmax(prediction,axis=1).flatten()==labels.flatten())/batch_size

test_accuracy_over_time=[]
train_loss_over_time=[]

for epoch in range(NUM_EPOCHS):
    train_loss=[]
    #训练阶段
    for step in range(train_inputs.shape[0]//batch_size):
        labels_one_hot=np.zeros((batch_size,num_labels),dtype=np.float32)
        labels_one_hot[np.arange(batch_size),train_labels[step*batch_size:(step+1)*batch_size]]=1.0

        #打印one-hot标签
        if epoch==0 and step==0:
            print('样例标签(one-hot)')
            print(labels_one_hot[:10])
            print()

        loss,_=session.run([tf_loss,tf_loss_minimize],feed_dict={train_inputs:train_inputs[step*batch_size:(step+1)*batch_size,:],
                                                          tf_labels:labels_one_hot})
        train_loss.append(loss)

        test_accuracy=[]
        for step in range(test_inputs.shape[0]//batch_size):
            test_predictions=session.run(tf_predictions,feed_dict={tf_inputs:test_inputs[step*batch_size:(step+1)*batch_size,:]})
            batch_test_accuracy=accuracy(test_predictions,test_labels[step*batch_size:(step+1)*batch_size])
            test_accuracy.append(batch_test_accuracy)

        print('第%d个epoch上训练数据的平均损失：%.3f\n'%(epoch+1,np.mean(train_loss)))
        train_loss_over_time.append(np.mean(train_loss))
        print('\t 第%d个epoch测试数据的平均准确率：%.2f\n'%(epoch+1,np.mean(test_accuracy)*100.0))
        test_accuracy_over_time.append(np.mean(test_accuracy)*100)
session.close()

import matplotlib.pyplot as plt
x_axis=np.arange(len(train_loss_over_time))
fig,ax=plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(w=25,h=5)
ax[0].plot(x_axis,train_loss_over_time)
ax[0].set_xlabel('Epochs',fontsize=18)
ax[0].set_ylabel('训练数据的平均损失',fontsize=18)
ax[0].set_title('训练数据的损失随时间的变化',fontsize=18)
ax[1].plot(x_axis,test_accuracy_over_time)
ax[1].set_xlabel('Epochs',fontsize=18)
ax[1].set_ylabel('测试数据的平均损失',fontsize=18)
ax[1].set_title('测试数据的准确率随时间的变化',fontsize=20)
plt.show()
