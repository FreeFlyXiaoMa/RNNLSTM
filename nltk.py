import theano
import theano.tensor as T
import numpy
from theano import function


x=T.dscalar('x')
y=T.dscalar('y')
z=x+y

f=function([x,y],z)
print(numpy.allclose(f(1,2),3))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#print(tf.__version__)
import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

tran_=np.genfromtxt('transfusion.csv',delimiter=',')
X=tran_[:,0:4]
y=tran_[:,4]
print(X)

mlp_keras=Sequential()
mlp_keras.add(Dense(8,input_dim=4,init='uniform',activation='relu'))
mlp_keras.add(Dense(6,init='uniform',activation='relu'))
mlp_keras.add(Dense(1,init='uniform',activation='sigmoid'))
mlp_keras.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

mlp_keras.fit(X,y,nb_epoch=20,batch_size=8,verbose=0)

accuracy=mlp_keras.evaluate(X,y)
print('accuracy=%.2f%%'%(accuracy[1]*100))



