# -*- coding: utf-8 -*-
#@Time    :2019/7/17 18:44
#@Author  :XiaoMa
import pickle
import tensorflow as tf
import os
import numpy as np

def unpickle_patch(file):
    patch_bin_file=open(file,'rb')
    patch_dict=pickle.load(patch_bin_file,encoding='bytes')

    return patch_dict

pathches_dir=''
def get_dataset_images(dataset_path,im_dim=32,num_channels=3):
    num_files=5
    images_per_file=10000
    files_names=os.listdir(pathches_dir)
    dataset_array=np.zeros(shape=(num_files*images_per_file,im_dim,im_dim,num_channels))
    dataset_labels=np.zeros(shape=(num_files*images_per_file),dtype=np.uint8)
    index=0
    for file_name in files_names:
        if file_name[0:len(file_name)-1]=='data_batch_':
            print('working on:',file_name)

            data_dict=unpickle_patch(dataset_path+file_name)
            images_data=data_dict[b'data']
            images_data_reshaped=np.reshape(images_data,newshape=(len(images_data),im_dim,im_dim,num_channels))
            dataset_array[index*images_per_file:(index+1)*images_per_file,:,:,:]=images_data_reshaped
            dataset_labels[index*images_per_file:(index+1)*images_per_file]=data_dict[b'labels']
            index=index+1
    return dataset_array,dataset_labels

def create_CNN(input_data,num_classes,keep_prop):
    filters,conv_layer1=create_conv_layer(input_data=input_data,filter_size=5,num_filters=4)
    relu_layer1=tf.nn.relu(conv_layer1)
    print('size of relu1 result:',relu_layer1.shape)

    max_pool_layer1=tf.nn.max_pool(value=relu_layer1,
                   ksize=[1,2,2,1],
                   strides=[1,1,1,1],
                   padding='VALID')
    print('size of maxpool result:',max_pool_layer1.shape)
    filters2,conv_layer2=create_conv_layer(input_data=max_pool_layer1,filter_size=7,num_filters=3)
    relu_layer2=tf.nn.relu(conv_layer2)
    print('size of relu2 reshape:',relu_layer2.shape)
    max_pool_layer2=tf.nn.max_pool(value=relu_layer2,
                   ksize=[1,2,2,1],
                   strides=[1,1,1,1],
                   padding='VALID')
    print('size of maxpool2 size:',max_pool_layer2.shape)

    filters3,conv_layer3=create_conv_layer(input_data=max_pool_layer2,filter_size=5,num_filters=2)
    relu_layer3=tf.nn.relu(conv_layer3)
    print('size of relu3 result:',relu_layer3.shape)
    max_pool_layer3=tf.nn.max_pool(value=relu_layer3,
                   ksize=[1,2,2,1],
                   strides=[1,1,1,1],
                   padding='VALID')
    print('size of maxpool3 result:',max_pool_layer3.shape)

    flattened_layer=dropout_flatten_layer(previous_layer=max_pool_layer3,keep_prop=keep_prop)
    fc_result1=fc_layer(flattened_layer=flattened_layer,num_inputs=flattened_layer.get_shape()[1:].num_elements(),num_output=200)
    fc_result2=fc_layer(flattened_layer=fc_result1,num_inputs=fc_result1.get_shape()[1:].num_elements(),
                        num_output=200)
    print('fully connected layer result:',fc_result2)
    return fc_result2

def create_conv_layer(input_data,filter_size,num_filters):
    filters=tf.Variable(tf.truncated_normal(shape=(filter_size,filter_size,tf.cast(input_data.shape[-1],dtype=tf.int32),num_filters),
                                    stddev=0.05))
    print('size of conv fiters bank:',filters.shape)
    conv_layer=tf.nn.conv2d(input=input_data,
                 filter=filters,
                 strides=[1,1,1,1],
                 padding='VALID')
    print('size of conv result:',conv_layer.shape)

    return filters,conv_layer

def dropout_flatten_layer(previous_layer,keep_prop):
    dropout=tf.nn.dropout(x=previous_layer,keep_prob=keep_prop)
    num_features=dropout.get_shape()[1:].num_elements()
    layer=tf.reshape(previous_layer,shape=(-1,num_features))
    return layer

def fc_layer(flattened_layer,num_inputs,num_output):
    fc_weights=tf.Variable(tf.truncated_normal(shape=(num_inputs,num_output),stddev=0.05))
    fc_result1=tf.matmul(flattened_layer,fc_weights)
    return fc_result1

num_dataset_classes=10
im_dim=32
num_channels=3
pathches_dir='tensorflow_demo/'
dataset_array,dataset_labels=get_dataset_images(dataset_path=pathches_dir,im_dim=im_dim,num_channels=num_channels)
print('size of data:',dataset_array.shape)

data_tensor=tf.placeholder(tf.float32,shape=[None,im_dim,im_dim,num_channels],name='data_tensor')
label_tensor=tf.placeholder(tf.float32,shape=[None,],name='label_tensor')

keep_prop=tf.Variable(initial_value=0.5,name='keep_prop')
fc_result2=create_CNN(input_data=data_tensor,num_classes=num_dataset_classes,keep_prop=keep_prop)
softmax_probabilities=tf.nn.softmax(fc_result2,name='softmax_probs')
softmax_predictions=tf.argmax(softmax_probabilities,axis=1)

cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=tf.reduce_max(input_tensor=softmax_probabilities,reduction_indices=[1]),
                                                      labels=label_tensor)
cost=tf.reduce_mean(cross_entropy)
error=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
sess=tf.Session()
tf.summary.FileWriter(logdir='tensorflow_demo/',graph=sess.graph)
sess.run(tf.global_variables_initializer())

def get_patch(data,labels,percent=70):
    num_elements=np.uint32(percent*data.shape[0]/100)
    shuffled_labels=labels
    np.random.shuffle(shuffled_labels)
    return data[shuffled_labels[:num_elements],:,:,:],shuffled_labels[:num_elements]


num_patches=5
for patch_num in np.arange(num_patches):
    print('Patch:',str(patch_num))
    percent=80
    shuffled_data,shuffled_labels=get_patch(data=dataset_array,labels=dataset_labels,percent=percent)

    cnn_feed_dict={
        data_tensor:shuffled_data,
        label_tensor:shuffled_labels,
        keep_prop:0.5
    }
    softmax_predictions_,_=sess.run([softmax_predictions,error],feed_dict=cnn_feed_dict)
    correct=np.array(np.where(softmax_predictions_==shuffled_labels))
    correct=correct.size
    print('correct predictions/',str(percent*5000/100),":",correct)

pathches_dir='tensorflow_demo/'
dataset_labels,dataset_labels=get_dataset_images(dataset_path=pathches_dir+'test_batch',im_dim=32,num_channels=3)
print('size of data:',dataset_array.shape)
sess=tf.Session()
saved_model_path='tensorflow_demo/'
saver=tf.train.import_meta_graph(saved_model_path+'model.ckpt.meta')
saver.restore(sess=sess,save_path=saved_model_path+'model.ckpt')

sess.run(tf.global_variables_initializer())
graph=tf.get_default_graph()

softmax_probabilities=graph.get_tensor_by_name(name='softmax_probs:0')
softmax_predictions=tf.argmax(softmax_probabilities,axis=1)



