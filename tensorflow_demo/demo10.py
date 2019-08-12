# -*- coding: utf-8 -*-
#@Time    :2019/8/12 15:05
#@Author  :XiaoMa
import tensorflow as tf

#网络层权重，n*m  n:上一层节点数  m:本层节点数
w1=tf.Variable(tf.random_normal([2,3],stddev=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1))

# 定义存放输入数据的地方，也就是x向量,这里shape为前一个传入训练的样本个数，后面出入每个样本的维度大小
x = tf.placeholder(tf.float32, shape=(None, 2), name="input")
# 矩阵乘法
a = tf.matmul(x,w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # 新版本好像不能用这个函数初始化所有变量了
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # feed_dict用于向y中的x传入参数，这里传入3个，则y输出为一个3*1的tensor
    print(sess.run(y,feed_dict={x:[[0.7,0.9],[1.0,1.5],[1,2]]}))




