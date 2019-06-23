# -*- coding: utf-8 -*-
#@Time    :2019/6/23 16:13
#@Author  :XiaoMa

import tensorflow as tf
a=tf.constant([1.0,2.0],name='a')
b=tf.constant([2.0,3.0],name='b')

#在计算图中读取变量的取值
"""with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable('v')))"""

g=tf.Graph()
with g.device('/gpu:0'):
    result=a+b
print(result)




