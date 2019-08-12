# -*- coding: utf-8 -*-
#@Time    :2019/8/9 15:52
#@Author  :XiaoMa

import tensorflow as tf
# w1=tf.Variable(tf.random_normal([2,3],stddev=1))
# w2=tf.Variable(tf.random_normal([3,1],stddev=1))
#
# #定义存放数据的地方，也就是x向量，这里shape为前一个传入训练的样本个数，后面输出每个样本的维度大小
# x=tf.placeholder(tf.float32,shape=(None,2),name='input')
# #矩阵乘法
# a=tf.matmul(x,w1)
# y=tf.matmul(a,w2)
#
# with tf.Session() as sess:
#     init_op=tf.initialize_all_variables()
#     sess.run(init_op)
#     #feed_dict用于向y中的x传入参数，这里传3个，则y输出为一个3*1的tensor
#     print(sess.run(y,feed_dict={x:[[0.7,0.9],[1.0,1.5],[2.1,2.3]]}))

# import numpy as np
# #导入数据，这里的数据是每一行代表一个样本，每一行最后一列表示样本标签
# data=np.loadtxt('trian_data.txt',dtype='float',delimiter=',')
#
# #将样本标签转换成独热编码
# def label_change(before_label):
#     label_num=len(before_label)
#     change_arr=np.zeros((label_num,33))
#     for i in range(label_num):
#         if before_label[i]==33.0:
#             change_arr[i,int(before_label[i]-1)]=1
#         else:
#             change_arr[i,int(before_label[i])]=1
#     return change_arr
# #定义神经网络的输入输出节点，每个样本为1*315维，以及输出分类结果
# INPUT_NODE=315
# OUTPUT_NODE=33
#
# #定义隐含层的神经网络，一层300个节点，一层100个节点
# LAYER1_NODE=300
# LAYER2_NODE=100
#
# #定义学习率，学习率衰减速度，正则系数，训练调整参数的次数以及平滑衰减率
# LEARNING_RATE_BASE=0.5
# LEARNING_RATE_DECAY=0.99
# REGULIZATION_RATE=0.0001
# TRAINING_STEPS=2000
# MOVING_AVERAGE_DECAY=0.99
#
# #定义整个神经网络的结构，也就是前向传播的过程，avg_class为平滑可训练的类，不传入则不使用平滑
# def inference(input_tensor,avg_class,w1,b1,w2,b2,w3,b3):
#     if avg_class==None:
#         #第一层 隐含层，输入与权重矩阵乘加后加上常数传入激活函数作为输出
#         layer1=tf.nn.relu(tf.matmul(input_tensor,w1)+b1)
#         #第二层隐含层，前一层的输出与权重矩阵相乘后加上常数作为输出
#         layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2)
#         return tf.matmul(layer2,w3)+b3
#     else:
#         #avg_class.average()平滑训练变量，也就是每一层与上一层的权重
#         layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(w1))+avg_class.average(b1))
#         layer2=tf.nn.relu(tf.matmul(layer1,avg_class.average(w2))+avg_class.average(b2))
#         return tf.matmul(layer2,avg_class.average(w3))+avg_class.average(b3)
#
# def train(data):
#     #清洗数据
#     np.random.shuffle(data)
#     #取前80个样本作为训练样本，后面的全是测试样本，约250个
#     data_train_x=data[:850,:315]
#     data_train_y=label_change(data[:850,-1])
#     data_test_x=data[850:,:315]
#     data_test_y=label_change(data[850:,-1])
#
#     #定义输出数据的地方，None表示无规定一次输入多少训练样本，y是样本标签存放的地方
#     x=tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
#     y_=tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y-input')
#
#     #依次定义每一层与上一层的权重，这里用随机数初始化，注意shape的对应关系
#     w1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER1_NODE],stddev=0.1))
#     b1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
#
#     w2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE,LAYER2_NODE],stddev=0.1))
#     b2=tf.Variable(tf.constant(0.1,shape=[LAYER2_NODE]))
#
#     w3=tf.Variable(tf.truncated_normal(shape=[LAYER2_NODE,OUTPUT_NODE],stddev=0.1))
#     b3=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
#
#     #输出向前传播的结果
#     y=inference(x,None,w1,b1,w2,b2,w3,b3)
#
#     #每训练完一次就会增加的变量
#     global_step=tf.Variable(0,trainable=False)
#     #定义平滑变量，输入为平滑衰减率和global_stop使得训练完一次就会使用平滑过程
#     variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
#     #将平滑应用到所有可训练的变量，即trainable=True的变量
#     variable_averages_op=variable_averages.apply(tf.trainable_variables())
#
#     #输出平滑后的预测值
#     average_y=inference(x,variable_averages,w1,b1,w2,b2,w3,b3)
#
#     #定义交叉熵和损失函数
#     cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
#
#     #计算交叉熵的平均值，也就是本轮训练对所有训练样本的平均值
#     cross_entrop_mean=tf.reduce_mean(cross_entropy)
#
#     #定义正则化权重，并将其加上交叉熵作为损失函数
#     regularizer=tf.contrib.layers.l2_regularizer(REGULIZATION_RATE)
#     regulization=regularizer(w1)+regularizer(w2)+regularizer(w3)
#     loss=cross_entrop_mean+regulization
#
#     #定义动态学习率，随着训练的步骤不断递减
#     learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,900,LEARNING_RATE_DECAY)
#     #定义向后传播的算法，梯度下降法，注意后面的Minimize要传入global_step
#     train_step=tf.learn.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#     #管理需要更新的变量，传入的参数是包含需要训练的变量的过程
#     train_op=tf.group(train_step,variable_averages_op)
#
#     #正确率预测
#     correct_prediction=tf.equal(tf.arg_max(average_y,1),tf.arg_max(y_,1))
#     accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
#     with tf.Session() as sess:
#         #初始所有变量
#         tf.global_variables_initializer().run()
#         #训练集输入字典
#         validate_feed={x:data_test_x,y_:data_train_y}
#         #测试集输入字典
#         test_feed={x:data_test_x,y_:data_test_y}
#
#         for i in range(TRAINING_STEPS):
#             if i%1000==0:
#                 validate_acc=sess.run(accuracy,feed_dict=validate_feed)
#                 print('After %d training steps,validation accuracy using average model is %g'%(i,validate_acc))
#             #每一轮通过同一训练集训练
#             sess.run(train_op,feed_dict=validate_feed)
#         #用测试集查看模型的准确率
#         test_acc=sess.run(accuracy,feed_dict=test_feed)
# train(data)
#
# from tensorflow import Session,constant,Variable,add,assign,initialize_all_variables
# state=Variable(0,name='counter')    #创建计算器
# one=constant(1)
# val=add(state,one)
# update=assign(state,val)    #更新变量
# setup=initialize_all_variables()#初始化Variable
# with Session() as session:
#     session.run(setup)  #执行初始化
#     print(session.run(state))   #输出初值
#     for i in range(3):
#         session.run(update) #执行更新
#         print(session.run(update))
#
# #动态地输入数据
# input1=tf.placeholder(tf.float32)
# input2=tf.placeholder(tf.float32)
#
# output=tf.matmul(input1,input2)
# with Session() as sess:
#     print(session.run(output,feed_dict={input1:[3],input2:[2]}))

"""
实现一个简单的神经网络
"""
class BPNeuralNetWork:
    def __init__(self):
        self.session=tf.Session()
        self.input_layer=None
        self.label_layer=None
        self.loss=None
        self.trianer=None
        self.layers=[]

    def __del__(self):
        self.session.close()

def make_layer(inputs,in_size,out_size,activate=None):
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biasis=tf.Variable(tf.zeros([1,out_size])+0.1)
    result=tf.matmul(inputs,weights)+biasis
    if activate is None:
        return result
    else:
        return activate(result)






