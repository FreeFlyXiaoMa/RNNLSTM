import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,Callback
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD,RMSprop,Adagrad,Adam
from keras.models import *
from keras.metrics import *
from keras.regularizers import  *
from keras.metrics import categorical_crossentropy

from keras_contrib.layers import CRF
#
max_len=1000
char_value_dict={}


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses=[]
    def on_batch_end(self, batch, logs=[]):
        self.losses.append(logs.get('loss'))
word_input=Input(shape=(max_len,),dtype='int32',name='word_input')
word_emb=Embedding(len(char_value_dict)+2,output_dim=64,input_length=max_len,
                   dropout=0.2,name='word_emb')(word_input)
bilstm=Bidirectional(LSTM(32,dropout_W=0.1,dropout_U=0.1,return_sequences=True))(word_emb)
bilstm_d=Dropout(0.1)(bilstm)
half_window_size=2
paddinglayer=ZeroPadding1D(padding=half_window_size)(word_emb)
conv=Conv1D(nb_filter=50,filter_length=(2*half_window_size+1),border_mode='valid')(word_emb)
conv_d=Dropout(0.1)(conv)
dense_conv=TimeDistributed(Dense(50))(conv)
rnn_cnn_merge=concatenate([bilstm_d,dense_conv])
class_label_count=10
dense=TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)
crf=CRF(class_label_count,sparse_target=False)
crf_output=crf(dense)
model=Model(input=[word_input],output=[crf_output])
model.compile(loss=crf.loss_function,optimizer='adam',metrics=[crf.accuracy])
model.summary()

#serialize model to JSON
model_json=model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)

#编译模型
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
#用于保存验证集误差最小的参数，当验证误差减少时，立马保存下来
checkpointer=ModelCheckpoint(filepath='bilstm_1102_k205_tf130.w',
                             verbose=0,save_best_only=True,
                             save_weights_only=True)
history=LossHistory()
model.fit(x=[],y=[],batch_size=32,epochs=500,
          callbacks=[checkpointer,history],
          validation_split=0.1)


