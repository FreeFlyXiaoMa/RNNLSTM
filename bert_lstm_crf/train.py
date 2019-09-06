# -*- coding: utf-8 -*-
#@Time    :2019/8/30 16:16
#@Author  :XiaoMa
import os
import pickle
from collections import OrderedDict
import tensorflow as tf

import pickle
import numpy as np
from bert_lstm_crf.model import Model
from bert_lstm_crf.utils import clean
from bert_lstm_crf.loader import load_sentences,tag_mapping,prepare_dataset
from bert_lstm_crf.data_util import BatchManager
from bert_lstm_crf.utils import make_path,load_config,save_config,get_logger,print_config,create_model


flags=tf.app.flags
flags.DEFINE_boolean('clean',False,'clean train folder')
flags.DEFINE_boolean('train',False,'whether train the model')
flags.DEFINE_integer('batch_size',128,'batch size')
flags.DEFINE_integer('seg_dim',20,'Embedding size for characters')
flags.DEFINE_integer('char_dim',100,'Embedding size for characters')
flags.DEFINE_integer('lstm_dim',200,'Num of hidden units in LSTM')
flags.DEFINE_string('tag_schema','iob','tagging schema iobes or iob')

#train
flags.DEFINE_float('clip',5,'Gradient Clip')
flags.DEFINE_float('dropout',0.5,'Dropout rate')
flags.DEFINE_float('lr',0.001,'Initial learning rate')
flags.DEFINE_string('optimizer','adam','Optimizer for training')
flags.DEFINE_boolean('zeros',False,'Wheter replace digits with zero')
flags.DEFINE_boolean('lower',False,'Wheter lower case')

flags.DEFINE_integer('max_seq_len',128,'max sequence length for bert')
flags.DEFINE_integer('max_epoch',10,'maximum training epochs')
flags.DEFINE_integer('step_check',100,'steps per check')
flags.DEFINE_string('ckpt_path','ckpt','path to save model')
flags.DEFINE_string('summary_path','summary','paths to store summaries')
flags.DEFINE_string('log_file','train.log','File for log')
flags.DEFINE_string('map_file','output/maps.pkl','file for maps')
flags.DEFINE_string('vocab_file','output/vocab.json','File for vocab')
flags.DEFINE_string('config_file','output/config_file','File for config')
flags.DEFINE_string('script','output/conlleval','evaluation script')
flags.DEFINE_string('result_path','result','paths for results')
flags.DEFINE_string('train_file',os.path.join('data','example.train'),'path for train data')
flags.DEFINE_string('dev_file',os.path.join('data','example.dev'),'path for dev data')
flags.DEFINE_string('test_file',os.path.join('data','example.test'),'path for test data')

FLAGS=tf.app.flags.FLAGS
assert FLAGS.clip<5.1 ,"gradient clip shoul't be too much"
assert 0<=FLAGS.dropout<1,'dropout rate between 0 and 1'
assert FLAGS.lr >0,'learning rate must larger than zero'
assert FLAGS.optimizer in ['adam','sgd','adagrad']


def config_model(tag_to_id):
    config=OrderedDict()
    config['num_tags']=len(tag_to_id)
    config['lstm_dim']=FLAGS.lstm_dim
    config['batch_size']=FLAGS.batch_size
    config['max_seq_len']=FLAGS.max_seq_len

    config['clip']=FLAGS.clip
    config['dropout_keep']=1.0-FLAGS.dropout
    config['optimizer']=FLAGS.optimizer
    config['lr']=FLAGS.lr
    config['tag_schema']=FLAGS.tag_schema
    config['zeros']=FLAGS.zeros
    config['lower']=FLAGS.lower

def train():
    train_sentences=load_sentences(FLAGS.train_file,FLAGS.lower,FLAGS.zeros)
    dev_sentences=load_sentences(FLAGS.dev_file,FLAGS.lower,FLAGS.zeros)
    test_sentences=load_sentences(FLAGS.test_file,FLAGS.lower,FLAGS.zeros)

    if not os.path.isfile(FLAGS.map_file):
        _t,tag_to_id,id_to_tag=tag_mapping(train_sentences)
        with open(FLAGS.map_file,'wb') as f:
            pickle.dump([tag_to_id,id_to_tag],f)
    else:
        with open(FLAGS.map_file,'rb') as f:
            tag_to_id,id_to_tag=pickle.load(f)
    train_data=prepare_dataset(train_sentences,FLAGS.max_seq_len,tag_to_id,FLAGS.lower)
    dev_data=prepare_dataset(dev_sentences,FLAGS.max_seq_len,tag_to_id,FLAGS.lower)
    test_data=prepare_dataset(test_sentences,FLAGS.max_seq_len,tag_to_id,FLAGS.lower)

    print('%i/%i/%i sentences in train/dev/test'%(len(train_data),0,len(test_data)) )
    train_manager=BatchManager(train_data,FLAGS.batch_size)
    dev_manager=BatchManager(dev_data,FLAGS.batch_size)
    test_manager=BatchManager(test_data,FLAGS.batch_size)

    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config=load_config(FLAGS.config_file)
    else:
        config=config_model(tag_to_id)
        save_config(config,FLAGS.config_file)
    make_path(FLAGS)

    log_path=os.path.join('log',FLAGS.log_file)
    logger=get_logger(log_path)
    print_config(config,logger)

    #limit GPU memory
    tf_config=tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    steps_per_epoch=train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model=create_model(sess,Model,FLAGS.ckpt_path,config,logger)
        logger.info('***start training***')
        loss=[]
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step,batch_loss=model.run_step(sess,True,batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check==0:
                    iteration=step//steps_per_epoch+1
                    logger.info('iteration:{} step"{}/{},NER loss:{:>9.6f}'.format(iteration,step%steps_per_epoch,steps_per_epoch,
                                                                                   np.mean(loss)))
            # best=
def evaluate(sess,model,name,data,id_to_tag,logger):
    logger.info('evaluate:{}'.format(name))
    ner_results=model.evaluate(sess,data,id_to_tag)



def main(_):
    FLAGS.train=True
    FLAGS.clean=True
    clean(FLAGS)
    train()

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    tf.app.run(main)






































