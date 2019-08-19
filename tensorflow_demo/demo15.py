# -*- coding: utf-8 -*-
#@Time    :2019/8/18 19:49
#@Author  :XiaoMa
import tensorflow as tf
# tf_x=tf.Variable(tf.constant(2,dtype=tf.float32))
# tf_y=tf_x**2
#
# minize_op=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(tf_y)
# print(minize_op)
sess=tf.InteractiveSession()
x=tf.Variable(tf.constant(2.0,dtype=tf.float32),name='x')
with tf.control_dependencies([tf.assign(x,x+5)]):
    z=x*2
tf.global_variables_initializer().run()
print('z=',sess.run(z))
print('x=',sess.run(x))
sess.close()











