#from imp import reload

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import math
import pickle

import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

#数据预处理
pickle_file='./notMNISTs.pickle'
with open(pickle_file,'rb') as f:
    save=pickle.load(f)
    train_dataset=save['train_dataset']
    train_labels=save['train_labels']
    valid_dataset=save['valid_dataset']
    valid_labels=save['valid_labels']
    test_dataset=save['test_dataset']
    test_labels=save['test_labels']
    del save
    print('Training set',train_dataset.shape[0],train_labels.shape)

#save=pickle.load(open('notMNISTs.pickle','r'),encoding='utf-8')

image_size=128
num_labels=10
def reformat(dataset,labels):
    dataset=dataset.reshape((-1,image_size*image_size)).astype(np.float)
    labels=(np.arange(num_labels)==labels[:,None]).astype(np.float)
    return dataset,labels
train_dataset,train_labels=reformat(train_dataset,train_labels)
valid_dataset,valid_labels=reformat(valid_dataset,valid_labels)
test_dataset,test_labels=reformat(test_dataset,test_labels)

#创建一个7层网络
layer_sizes=[784,1000,500,250,250,250,10]
L=len(layer_sizes)-1
num_examples=train_dataset.shape[0]
num_epochs=100
start_learning_rate=0.02
decay_after=15  #epoch after wich to begin learning rate decay
batch_size=120
num_iter=(num_examples/batch_size)*num_epochs

x=tf.placeholder(tf.float32,shape=(None,layer_sizes[0]))
outputs=tf.placeholder(tf.float32)
testing=tf.placeholder(tf.bool)
learning_rate=tf.Variable(start_learning_rate,trainable=False)

def bi(inits,size,name):
    return tf.Variable(inits*tf.ones([size]),name=name)

def wi(shape,name):
    return tf.Variable(tf.random_normal(shape,name=name))/math.sqrt(shape[0])
shapes=zip(layer_sizes[:-1],layer_sizes[1:])

weights={'W':[wi(s,"W") for  s in shapes],
         'beta':[bi(0.0,layer_sizes[l+1],'beta') for l in range(L)],
         'gamma':[bi(1.0,layer_sizes[l+1],'beta') for l in range(L)]
         }

#to calculate the moving averages of mean and variance
ewma=tf.train.ExponentialMovingAverage(decay=0.99)
bn_assigns=[] #this list stores the update to be made to average mean and variance

def batch_normalization(batch,mean=None,var=None):
    if mean is None or var is None:
        mean,var=tf.nn.moments(batch,axes=[0])
    return (batch-mean)/tf.sqrt(var+tf.constant(1e-10))

#average mean and variance of all layers
running_mean=[tf.Variable(tf.constant(0.0,shape=[l]),trainable=False) for l in layer_sizes[1:]]
running_var=[tf.Variable(tf.constant(1.0,shape=[l]),trainable=False) for l in layer_sizes[1:]]

def update_batch_normalization(batch,l):
    mean,var=tf.nn.moments(batch,axes=[0])
    assign_mean=running_mean[l-1].assign(mean)
    assign_var=running_var[l-1].assign(var)
    bn_assigns.append(ewma.apply([running_mean[l-1],running_var[l-1]]))
    with tf.control_dependencies([assign_mean,assign_var]):
        return (batch-mean)/tf.sqrt(var+1e-10)



