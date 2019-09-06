# -*- coding: utf-8 -*-
#@Time    :2019/9/2 15:15
#@Author  :XiaoMa
import codecs
from bert_lstm_crf.data_util import zero_digits,create_dico,create_mapping
from bert_lstm_crf.bert import tokenization
from bert_lstm_crf.utils import convert_single_example



tokenizer=tokenization.FullTokenizer(vocab_file='chinese_L-12_H-768_A-12/vocab.txt',
                                     do_lower_case=True)


def load_sentences(path,lower,zeros):
    """
    load sentences, a line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    :param path:
    :param lower:
    :param zeros:
    :return:
    """
    sentences=[]
    sentence=[]
    num=0
    for line in codecs.open(path,'r','utf-8'):
        num+=1
        line=zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence)>0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence=[]
        else:
            if line[0]==' ':
                line="$"+line[1:]
                word=line.split()
            else:
                word=line.split()
            assert len(word)>=2,print([word[0]])
            sentence.append(word)

    if len(sentence)>0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def tag_mapping(sentences):
    """
    create a dictionary and a mapping of tags,sorted by frequency
    :param sentences:
    :return:
    """
    tags=[[s[-1] for s in char] for char in sentences]
    dico=create_dico(tags)
    dico['[SEP]']=len(dico)+1
    dico['[CLS]']=len(dico)+2

    tag_to_id,id_to_tag=create_mapping(dico)
    print('Found %d unique named entity tags'%len(dico))
    return dico,tag_to_id,id_to_tag

def prepare_dataset(sentences,max_seq_length,tag_to_id,lower=False,train=True):
    """
    prepare the dataset.Return a list of lists of dictionaries containing:
    -word indexes
    -word char indexes
    -tag indexes
    :param sentences:
    :param max_seq_length:
    :param tag_to_id:
    :param lower:
    :param train:
    :return:
    """
    def f(x):
        return x.lower() if lower else x
    data=[]
    for s in sentences:
        string=[w[0].strip() for w in s]
        char_line=' '.join(string)
        text=tokenization.convert_to_unicode(char_line)
        if train:
            tags=[w[-1] for w in s]
        else:
            tags=['0' for _ in string]
        labels=' '.join(tags)
        labels=tokenization.convert_to_unicode(labels)
        ids,mask,segment_ids,label_ids=convert_single_example(char_line=text,
                               tag_to_id=tag_to_id,
                               max_seq_length=max_seq_length,
                               tokenizer=tokenizer,
                               label_line=labels)
        data.append([string,segment_ids,ids,mask,label_ids])
    return data
