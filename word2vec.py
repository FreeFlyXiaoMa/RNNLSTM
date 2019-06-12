import random
import collections
import math
import os
import zipfile
import time
import re
import numpy as np
import tensorflow as tf

from six.moves import range
from six.moves.urllib.request import urlretrieve

dataset_link='http://mattmahoney.net/dc/'
zip_file='text8.zip'

def data_download(zip_file):
    '''download the required file'''
    if not os.path.exists(zip_file):
        zip_file,_=urlretrieve(dataset_link+zip_file,filename=zip_file)
        print('file downloaded successfully!')
    return None

data_download(zip_file)

'''extracting the dataset in separate folder'''
extracted_folder='dataset'
if not os.path.isdir(extracted_folder):
    with zipfile.ZipFile(zip_file,mode='r') as zf:
        zf.extractall(extracted_folder)
with open('dataset/text8') as ft_:
    full_text=ft_.read()

def text_processing(ft8_text):
    '''replacing punctuation marks with tokens'''
    ft8_text=ft8_text.lower()
    ft8_text=ft8_text.replace('.','<period>')
    ft8_text=ft8_text.replace(',','<comma>')
    ft8_text=ft8_text.replace('"','<quotation>')
    ft8_text=ft8_text.replace(';','<semicolon>')
    ft8_text=ft8_text.replace('!','<exclamation>')
    ft8_text=ft8_text.replace('?','<question>')
    ft8_text=ft8_text.replace('(','<paren_l>')
    ft8_text=ft8_text.replace(')','<paren_r>')
    ft8_text=ft8_text.replace('--','<hyphen>')
    ft8_text=ft8_text.replace(':','<colon>')
    ft8_text_tokens=ft8_text.split()
    return ft8_text_tokens
ft_tokens=text_processing(full_text)

'''shortlisting words with frequency more than 7'''
word_cnt=collections.Counter(ft_tokens)
shortlisted_words=[w for w in ft_tokens if word_cnt[w]>7]

#列出数据集中词频最高的几个单词，如下所示
print(shortlisted_words[:15])

#检查数据集中所有单词的统计信息
print("Totalnumber of shorted listed words:",len(shortlisted_words))
print("Unique number of shortlisted words:",len(set(shortlisted_words)))

"""
The function creates a dictionary of words present in dataset 
alonog with their frequency order
"""
def dict_creation(shortlisted_words):
    counts=collections.Counter(shortlisted_words)
    vocabulary=sorted(counts,key=counts.get,reverse=True)
    rev_dictionary={ii:word for ii,word in enumerate(vocabulary)}
    dictionary_={word:ii for ii,word in rev_dictionary.items()}
    return dictionary_,rev_dictionary
words_cnt=[dict_creation(word) for word in shortlisted_words]

