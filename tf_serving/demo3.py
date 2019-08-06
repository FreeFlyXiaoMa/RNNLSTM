# -*- coding: utf-8 -*-
#@Time    :2019/8/6 16:04
#@Author  :XiaoMa

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
#设置模型训练时，输出日志信息
tf.logging.set_verbosity(tf.logging.INFO)

print('Using Tensorflow version %s\n'%(tf.__version__))

CONTINUOUS_COLUMNS=["I"+str(i) for i in range(1,14)]
CATEGORICAL_COLUMNS=['C'+str(i) for i in range(1,27)]
LABEL_COLUMN=['clicked']

TRAIN_DATA_COLUMNS=LABEL_COLUMN+CONTINUOUS_COLUMNS+CATEGORICAL_COLUMNS

FEATURE_COLUMNS=CONTINUOUS_COLUMNS+CATEGORICAL_COLUMNS

# print('Feature columns are:',FEATURE_COLUMNS)

sample = [ 0 , 2, 11, 5, 10262, 34, 2, 4, 5,0 , 1,0 , 5,
           "be589b51", "287130e0", "cd7a7a22", "fb7334df", "25c83c98","0" , "6cdb3998", "361384ce",
           "a73ee510", "3ff10fb2", "5874c9c9", "976cbd4c", "740c210d", "1adce6ef", "310d155b", "07eb8110",
           "07c540c4", "891589e7", "18259a83", "a458ea53", "a0ab60ca","0" , "32c7478e", "a052b1ed",
           "9b3e8820", "8967c0d2"]

# print('Columns and data as a dict:',dict(zip(FEATURE_COLUMNS,sample)),'\n')
BATCH_SIZE=400

def generate_input_fn(filename,batch_size=BATCH_SIZE):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        # Reads out batch_size number of lines
        key, value = reader.read_up_to(filename_queue, num_records=batch_size)

        # 1 int label, 13 ints, 26 strings
        cont_defaults = [[0] for i in range(1, 14)]
        cate_defaults = [[" "] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        # The label is the first column of the data.
        record_defaults = label_defaults + cont_defaults + cate_defaults

        # Decode CSV data that was just read out.
        # Note that this does NOT return a dict,
        # so we will need to zip it up with our headers
        columns = tf.decode_csv(
            value, record_defaults=record_defaults)

        # all_columns is a dictionary that maps from column names to tensors of the data.
        all_columns = dict(zip(column_headers, columns))

        # Pop and save our labels
        # dict.pop() returns the popped array of values; exactly what we need!
        labels = all_columns.pop(LABEL_COLUMN[0])

        # the remaining columns are our features
        features = all_columns

        # Sparse categorical features must be represented with an additional dimension.
        # There is no additional work needed for the Continuous columns; they are the unaltered columns.
        # See docs for tf.SparseTensor for more info
        for feature_name in CATEGORICAL_COLUMNS:
            features[feature_name] = tf.expand_dims(features[feature_name], -1)

        return features, labels
    return _input_fn
print('input function configured!')

wide_columns=[]
for name in CATEGORICAL_COLUMNS:
    wide_columns.append(tf.contrib.layers.sparse_column_with_hash_bucket(
                        name,hash_bucket_size=1000
    ))
print('wide/sparse columns configured')

deep_columns=[]
for name in CONTINUOUS_COLUMNS:
    deep_columns.append(tf.contrib.layers.real_valued_column(name))
print('deep/contimuous columns configured!')

# for col in wide_columns:
#     deep_columns.append(tf.contrib.layers.embedding_column(col,dimension=8))
# print('wide and deep columns configured!')

def create_model_dir(model_type):
    return 'models/model_'+model_type+'_'+str(int(time.time()))

def get_model(model_type,model_dir):
    print('Model directory=%s'%(model_dir))

    runconfig=tf.contrib.learn.RunConfig(save_checkpoints_secs=None,save_checkpoints_steps=100)

    m=None

    #Linear Classifier
    if model_type=='WIDE':
        m=tf.contrib.learn.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns
        )
    #deep neural net classifier
    if model_type=='DEEP':
        m=tf.contrib.learn.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns
        )
    if model_type=='WIDE_AND_DEEP':
        m=tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100,70,50,25],
            config=runconfig
        )
    return m
MODEL_TYPE='WIDE_AND_DEEP'
model_dir=create_model_dir(model_type=MODEL_TYPE)
m=get_model(MODEL_TYPE,model_dir)
from tensorflow.contrib.learn.python.learn import evaluable
isinstance(m,evaluable.Evaluable)
print('开始训练模型！')
#训练模型
# CLOUD
train_file = "./train.csv"
eval_file  = "./eval.csv"

train_sample_size=800000
train_steps=train_sample_size/BATCH_SIZE
m.fit(input_fn=generate_input_fn(train_file,BATCH_SIZE),steps=train_steps)
print('fit done!')

#模型评估
eval_sample_size=200000
eval_steps=eval_sample_size/BATCH_SIZE
results=m.evaluate(input_fn=generate_input_fn(eval_file),steps=eval_steps)
print('evaluate done')
print('Accuracy:%s'%results['accuracy'])
print(results)

#模型预测
def pred_fn():
    sample = [0, 127, 1, 3, 1683, 19, 26, 17, 475, 0, 9, 0, 3, "05db9164", "8947f767", "11c9d79e", "52a787c8",
              "4cf72387", "fbad5c96", "18671b18", "0b153874", "a73ee510", "ceb10289", "77212bd7", "79507c6b",
              "7203f04e", "07d13a8f", "2c14c412", "49013ffe", "8efede7f", "bd17c3da", "f6a3e43b", "a458ea53",
              "35cd95c9", "ad3062eb", "c7dc6720", "3fdb382b", "010f6491", "49d68486"]
    sample_dict = dict(zip(FEATURE_COLUMNS, sample))
    # print('sample_dict:',sample_dict)

    for feature_name in CATEGORICAL_COLUMNS:
        sample_dict[feature_name] = tf.expand_dims(sample_dict[feature_name], -1)

    for feature_name1 in CONTINUOUS_COLUMNS:
        sample_dict[feature_name1] = tf.constant(sample_dict[feature_name1],shape=None, dtype=tf.int32)
    # print(sample_dict)

    return sample_dict
value=m.predict(input_fn=pred_fn)
print('预测结果：',value)
