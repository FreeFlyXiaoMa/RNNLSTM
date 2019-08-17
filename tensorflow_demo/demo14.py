# -*- coding: utf-8 -*-
#@Time    :2019/8/18 0:09
#@Author  :XiaoMa
import tensorflow as tf
import numpy as np
"""
tensorflow张量例子
"""
graph=tf.Graph()
session=tf.InteractiveSession(graph=graph)

x=tf.placeholder(dtype=tf.float32,shape=[1,10],name='x')
W=tf.Variable(tf.random_uniform(shape=[10,5],minval=-0.1,maxval=0.1,dtype=tf.float32),name='w')
b=tf.Variable(tf.zeros(shape=[5],dtype=tf.float32),name='b')

h=tf.nn.sigmoid(tf.matmul(x,W)+b)

tf.global_variables_initializer().run()
session.run(h,feed_dict={x:np.random.rand(1,10)})
session.close()



