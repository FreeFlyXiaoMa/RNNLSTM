# -*- coding: utf-8 -*-
#@Time    :2019/8/12 15:22
#@Author  :XiaoMa
import tensorflow as tf
import numpy as np
data=np.loadtxt('./tf_train_data.txt',dtype='float',delimiter=',')

#标签转换为独热编码
def label_change(befor_label):
    label_num=len(befor_label)
    change_arr=np.zeros((label_num,33))
    for i in range(label_num):
        if befor_label[i]==33.0:
            change_arr[i,int(befor_label[i]-1)]=1
        else:
            change_arr[i,int(befor_label[i])]=1
    return change_arr

#定义神经网络的输入输出节点，样本维度为315，输出分类结果
INPUT_NODE=315
OUTPUT_NODE=33
#定义两层隐含层的神经网络，一层300个节点，一层100个节点
LAYER1_NODE=300
LAYER2_NODE=100

#学习率、学习率衰减速度，正则系数，训练调整参数的次数，平滑衰减率
LEARNING_RATE_BASE=0.5
LEARNING_RATE_DECAY=0.99
REGULIZATION_RATE=0.0001
TRAING_STEPS=2000
MOVING_AVERAGE_DECAY=0.99

#定义神经网络的结构
def inference(input_tensor,avg_class,w1,b1,w2,b2,w3,b3):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,w1)+b1)
        layer2=tf.nn.relu(tf.matmul(layer1,avg_class.average(w2))+avg_class.average(b2))

        return tf.nn.relu(tf.matmul(layer2,avg_class.average(w3))+avg_class.average(b3))
def train(data):
    #打乱数据
    np.random.shuffle(data)
    data_train_x=data[:850,:315]
    data_train_y=label_change(data[:850,-1])
    data_test_x=data[850:,:314]
    data_test_y=label_change(data[850:,-1])

    #定义输出数据的地方
    x=tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='input')
    y_=tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='output')

    #定义每一层与上一层的权重，
    w1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER1_NODE],stddev=0.1))
    b1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    w2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE,LAYER2_NODE],stddev=0.1))
    b2=tf.Variable(tf.constant(0.1,shape=[LAYER2_NODE]))

    w3=tf.Variable(tf.truncated_normal(shape=[LAYER2_NODE,OUTPUT_NODE],stddev=0.1))
    b3=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    y=inference(x,None,w1,b1,w2,b2,w3,b3)
    #每训练完一次就会增加的变量
    global_step=tf.Variable(0,trainable=False)
    #定义平滑变量的类，输入为平滑衰减率和global_stop使得每训练完一次就会使用平滑过程
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    #输出平滑后的预测值
    average_y=inference(x,variable_averages,w1,b1,w2,b2,w3,b3)

    #交叉熵和损失函数
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
    cros_entripy_mean=tf.reduce_mean(cross_entropy)

    #定义正则化权重，并将其加上交叉熵作为损失函数
    regulizer=tf.contrib.layers.l2_regularizer(REGULIZATION_RATE)
    regularization=regulizer(w1)+regulizer(w2)+regulizer(w3)

    loss=cross_entropy+regularization

    #定义动态学习率，随着训练的步骤增加不断递减
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,900,LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    #管理需要更细的变量，传入的参数是包含需要训练的变量的过程
    train_op=tf.group(train_step,variable_averages_op)

    #真确率预测
    correct_prediction=tf.equal(tf.arg_max(average_y,1),tf.arg_max(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #训练集插入字典
        validate_feed={x:data_train_x,y:data_train_y}
        #测试集输入字典
        test_feed={x:data_test_x,y:data_test_y}

        for i in range(TRAING_STEPS):
            if i %1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print('After %d training steps,validation steps accuracy using average model is %g'%(i,validate_acc))
            #每一轮通过同一训练集训练
            sess.run(train_op,feed_dict=validate_feed)
        #用测试集查看模型的准确率
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print('After %d training steps,test accuracy using average model is %g'%(TRAING_STEPS,test_acc))
train(data)






