# -*- coding: utf-8 -*-
#@Time    :2019/7/31 15:03
#@Author  :XiaoMa
import tensorflow as tf

w=tf.Variable([.3],dtype=tf.float32)
b=tf.Variable([-.3],dtype=tf.float32)

#声明变量
x=tf.placeholder(tf.float32)
linear_model=w*x+b
#声明变量
y=tf.placeholder(tf.float32)

#loss损失函数，相减的平方再求和
loss=tf.reduce_sum(tf.square(y-linear_model))

#optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

#train data
x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]

#训练过程
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)  #参数初始化
for i in range(1000):
    sess.run(train,{x:x_train,y:y_train})

#evulate training accuracy
curr_W,curr_b,curr_loss=sess.run([w,b,loss],{x:x_train,y:y_train})
print('W:%s b:%s loss:%s'%(curr_W,curr_b,curr_loss))




















