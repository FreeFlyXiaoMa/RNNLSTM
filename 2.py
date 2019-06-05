from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671) #重复性设置
#网络和训练
NB_EPOCH=250
BATCH_SIZE=128
VERBOSE=1
NB_CLASSES=10
OPTIMIZER=SGD()
N_HIDDEN=128
VALIDATION_SPLIT=0.2
DROPOUT=1.3

#数据：混合并划分训练集和测试集数据
(X_train,y_train),(X_test,y_test)=mnist.load_data()
RESHAPED=784

X_train=X_train.reshape(60000,RESHAPED)
X_test=X_test.reshape(60000,RESHAPED)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

#归一化
X_train/=255
X_test/=255

#将类向量转换为二值类别矩阵
y_train=np_utils.to_categorical(y_train,NB_CLASSES)
y_test=np_utils.to_categorical(y_test,NB_CLASSES)

#N_HIDDEN个隐藏层，10个输出
model=Sequential()
model.add(Dense(N_HIDDEN,input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_entropy',optimizer=OPTIMIZER,metrics=['accuracy'])
history=model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)
score=model.evaluate(X_test,y_test,verbose=VERBOSE)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])




