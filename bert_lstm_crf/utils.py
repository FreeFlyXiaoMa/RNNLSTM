# -*- coding: utf-8 -*-
#@Time    :2019/9/2 14:52
#@Author  :XiaoMa
import os
import shutil
import tensorflow as tf

def clean(params):
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)
    if os.path.isdir(params.ckpt_path):
        os.remove(params.ckpt_path)
    if os.path.isdir(params.summary_path):
        os.remove(params.summary_path)
    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)
    if os.path.isdir('log'):
        shutil.rmtree('log')
    if os.path.isdir('__pycache__'):
        shutil.rmtree('__pycache__')

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

def convert_single_example(char_line,tag_to_id,max_seq_length,tokenizer,label_line):
    """
    将一个样本进行分析，将字转换为id，将label转换为id
    :param char_line:
    :param tag_to_id:
    :param max_seq_length:
    :param tokenizer:
    :param label_line:
    :return:
    """
    text_list=char_line.split(' ')
    label_list=label_line.split(' ')

    tokens=[]
    labels=[]
    for i,word in enumerate(text_list):
        token=tokenizer.tokenize(word)
        tokens.extend(token)
        label_l=label_list[i]
        for m in range(len(token)):
            if m==0:
                labels.append(label_l)
            else:
                labels.append('X')
        #序列截断
        if len(tokens) >=max_seq_length-1:
            tokens=tokens[0:(max_seq_length-2)]
            labels=labels[0:(max_seq_length-2)]
        ntokens=[]
        segment_ids=[]
        label_ids=[]
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(tag_to_id["[CLS]"])
        for i,token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(tag_to_id[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(tag_to_id['[SEP]'])
        input_ids=tokenizer.convert_tokens_to_ids(ntokens)
        input_mask=[1]*len(input_ids)

        #padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            #
            label_ids.append(0)
            ntokens.append("***NULL***")
        return input_ids,input_mask,segment_ids,label_ids


def make_path(params):
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir('log'):
        os.makedirs('log')

import json
def load_config(config_file):
    with open(config_file,'r',encoding='utf-8') as f:
        json.load(f)

def save_config(config,config_file):
    with open(config_file,'w') as f:
        json.dump(config,f,ensure_ascii=False,indent=4)

import logging
def get_logger(log_file):
    logger=logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh=logging.FileHandler(log_file)
    ch=logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
def print_config(config,logger):
    """
    print configuration of the model
    :param config:
    :param logger:
    :return:
    """
    for k,v in config.items():
        logger.info("{}:\t{}.".format(k.ljust(15),v))


def create_model(session,Model_class,path,config,logger):
    model=Model_class(config)
    ckpt=tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info('Reading model parameters from %s'%ckpt.model_checkpoint_path)
        model.saver.restore(session,ckpt.model_checkpoint_path)
    else:
        session.run(tf.global_variables_initializer())
    return model

import codecs
def test_ner(results,path):
    output_file=os.path.join(path,'ner_predict.utf8')
    with codecs.open(output_file,'w','utf-8') as f:
        to_write=[]
        for block in results:
            for line in block:
                to_write.append(line+'\n')
            to_write.append('\n')
        f.writelines(to_write)

# def return_report(input_file):
#     with codecs.open(input_file,'r','utf-8') as f:

