# -*- coding: utf-8 -*-
#@Time    :2019/8/30 16:17
#@Author  :XiaoMa
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
from bert_lstm_crf.bert import modeling
# from tensorflow.contrib.rnn import rnn
from bert_lstm_crf import rnncell as rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import numpy as np

class Model(object):
    def __init__(self,config):
        self.config=config
        self.lr=config['lr']
        self.lstm_dim=config['lstm_dim']
        self.num_tags=config['num_tags']

        self.global_step=tf.Variable(0,trainable=False)
        self.best_dev_f1=tf.Variable(0,trainable=False)
        self.best_test_f1=tf.Variable(0,trainable=False)
        self.initializer=initializers.xavier_initializer()

        self.input_ids=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_ids')
        self.input_mask=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_mask')
        self.segment_ids=tf.placeholder(dtype=tf.int32,shape=[None,None],name='segment_ids')
        self.targets=tf.placeholder(dtype=tf.int32,shape=[None,None],name='Targets')
        self.dropout=tf.placeholder(dtype=tf.int32,shape=[None,None],name='Dropout')

        used=tf.sign(tf.abs(self.input_ids))
        length=tf.reduce_sum(used,reduction_indices=1)
        self.lengths=tf.cast(length,tf.int32)
        self.batch_size=tf.shape(self.input_ids)[0]
        self.num_steps=tf.shape(self.input_ids)[-1]

        embedding=self.bert_embedding()
        lstm_inputs=tf.nn.dropout(embedding,self.dropout)
        lstm_outputs=self.biLSTM_layer(lstm_inputs,self.lstm_dim,self.lengths)

        self.logits=self.project_layer(lstm_outputs)

        self.loss=self.loss_op(self.logits,self.lengths)

        #bert模型参数初始化
        init_checkpoint='chinese_L-12_H-768_A-12/bert_model.ckpt'
        tvars=tf.trainable_variables()
        #加载Bert模型
        (assignment_map,initialized_variable_names)=modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint,assignment_map)
        print('*** Trainable Variables ***')
        #打印加载模型的参数
        train_vars=[]
        for var in tvars:
            init_string=''
            if var.name in initialized_variable_names:
                init_string=',*INIT_FROM_CKPT'
            else:
                train_vars.append(var)
            print(' name=%s, shape=%s%s',var.name,var.shape,init_string)
        with tf.variable_scope('optimizer'):
            optimizer=self.config['optimizer']
            if optimizer=='adam':
                self.opt=tf.train.AdamOptimizer(self.lr)
            else:
                raise KeyError

            grads=tf.gradients(self.loss,train_vars)
            (grads,_)=tf.clip_by_global_norm(grads,clip_norm=1.0)
            self.train_op=self.opt.apply_gradients(zip(grads,train_vars),global_step=self.global_step)
        self.saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)



    def biLSTM_layer(self,lstm_inputs,lstm_dim,lengths,name=None):
        """

        :param lstm_inputs:
        :param lstm_dim:
        :param length:
        :param name:
        :return:
        """
        with tf.variable_scope('char_BiLSTM' if not name else name):
            lstm_cell={}
            for direction in ['forward','backword']:
                lstm_cell[direction]=rnn.CoupledInputForgetGateLSTMCell(
                    lstm_dim,
                    use_peepholes=True,
                    initializer=self.initializer,
                    state_is_tuple=True
                )
            outputs,final_states=tf.nn.bidirectional_dynamic_rnn(
                lstm_cell['forward'],
                lstm_cell['backword'],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths
            )
        return tf.concat(outputs,axis=2)

    def project_layer(self,lstm_outputs,name=None):
        """

        :param lstm_outputs:
        :param name:
        :return:
        """
        with tf.variable_scope('project' if not name else name):
            with tf.variable_scope('hidden'):
                W=tf.get_variable('W',shape=[self.lstm_dim*2,self.lstm_dim],
                              dtype=tf.float32,initilizer=self.initializer)
                b=tf.get_variable('b',shape=[self.lstm_dim],dtype=tf.float32,initializer=tf.zeros_initializer())
                output=tf.reshape(lstm_outputs,shape=[-1,self.lstm_dim*2])
                hidden=tf.tanh(tf.nn.xw_plus_b(output,W,b))
            with tf.variable_scope('logits'):
                W=tf.get_variable('W',shape=[self.lstm_dim,self.num_tags],dtype=tf.float32,initializer=self.initializer)
                b=tf.get_variable('b',shape=[self.num_tags],dtype=tf.float32,initializer=tf.zeros_initializer())

                pred=tf.nn.xw_plus_b(hidden,W,b)
        return tf.reshape(pred,shape=[-1,self.num_tags,self.num_tags])

    def bert_embedding(self):
        bert_config=modeling.BertConfig.from_json_file('chinese_L-12_H-768_A-12/bert_config.json')
        model=modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False

        )
        embedding=model.get_sequence_output()
        return embedding

    def loss_op(self,project_logits,lengths,name=None):
        with tf.variable_scope('crf_loss' if not name else name):
            small=-1000.0
            start_logits=tf.concat([small*tf.ones(shape=[self.batch_size,1,self.num_tags]), tf.zeros(shape=[self.batch_size,1,1])],axis=-1)
            pad_logits=tf.cast(small*tf.ones([self.batch_size,self.num_tags,1]),tf.float32)
            logits=tf.concat([project_logits,pad_logits],axis=-1)
            logits=tf.concat([start_logits,logits],axis=1)
            targets=tf.concat([tf.cast(self.num_tags*tf.ones([self.batch_size,1]),tf.int32),self.targets],axis=-1)
            self.trans=tf.get_variable('transitions',shape=[self.num_tags+1,self.num_tags+1],initializer=self.initializer)
            log_likelihood,self.trans=crf_log_likelihood(inputs=logits,tag_indices=targets,transition_params=self.trans,sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self,is_train,batch):
        _,segment_ids,chars,mask,tags=batch
        feed_dict={
            self.input_ids:np.asarray(chars),
            self.input_mask:np.asarray(mask),
            self.segment_ids:np.asarray(segment_ids),
            self.dropout:1.0
        }
        if is_train:
            feed_dict[self.targets]=np.asarray(tags)
            feed_dict[self.dropout]=self.config['dropout_keep']
        return feed_dict

    def run_step(self,sess,is_train,batch):
        feed_dict=self.create_feed_dict(is_train,batch)
        if is_train:
            global_step,loss,_=sess.run([self.global_step,self.loss,self.train_op],feed_dict)
            return global_step,loss
        else:
            lengths,logits=sess.run([self.lengths,self.logits],feed_dict)
            return lengths,logits
    def decode(self,logits,lengths,matrix):
        paths=[]
        small=-1000.0
        start=np.asarray([[small]*self.num_tags+[0]])
        for score,length in zip(logits,lengths):
            score=score[:length]
            pad=small*np.ones([length,1])
            logits=np.concatenate([score,pad],axis=1)
            logits=np.concatenate([start,logits],axis=0)
            path,_=viterbi_decode(logits,matrix)
            paths.append(path[1:])
        return paths

    def evaluate(self,sess,data_manager,id_to_tag):
        """

        :param sess:
        :param data_manager:
        :param id_to_tag:
        :return:
        """
        results=[]
        trans=self.trans.eval()
        for batch in data_manager.iter_batch():
            strings=batch[0]
            labels=batch[-1]
            lenghts,scores=self.run_step(sess,False,batch)
            batch_paths=self.decode(scores,lenghts,trans)
            for i in range(len(strings)):
                result=[]
                string=strings[i][:lenghts[i]]
                gold=[id_to_tag[int(x)] for x in labels[i][1:lenghts[i]]]
                pred=[id_to_tag[int(x)] for x in batch_paths[i][1:lenghts[i]]]
                for char,gold,pred in zip(string,gold,pred):
                    result.append(' '.join([char,gold,pred]))
                results.append(result)
        return results
