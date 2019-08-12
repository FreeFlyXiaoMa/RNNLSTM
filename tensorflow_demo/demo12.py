# -*- coding: utf-8 -*-
#@Time    :2019/8/12 16:59
#@Author  :XiaoMa
from __future__ import absolute_import,division,print_function

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging_ERROR)
import matplotlib.pyplot as plt

import numpy as np
celsius_q=np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit_a=np.array([-40,14,32,46,59,72,100],dtype=float)

l0=tf.keras.layers.Dense(units=4,input_shape=[1])
l1=tf.keras.layers.Dense(units=4)
l2=tf.keras.layers.Dense(units=1)
model=tf.keras.Sequential([l0,l1,l2])
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
history=model.fit(celsius_q,fahrenheit_a,epochs=45,verbose=False)
print('Finished train!')
print('预测结果：',model.predict([100.0]))
print('l0 weights:',l0.get_weights())
print('l1 weights:',l1.get_weights())
print('l2 weights:',l2.get_weights())

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.show()

