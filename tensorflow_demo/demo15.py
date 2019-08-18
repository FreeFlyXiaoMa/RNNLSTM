# -*- coding: utf-8 -*-
#@Time    :2019/8/18 19:49
#@Author  :XiaoMa
import tensorflow as tf
tf_x=tf.Variable(tf.constant(2,dtype=tf.float32))
tf_y=tf_x**2

minize_op=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(tf_y)
print(minize_op)













