# -*- coding: utf-8 -*-
#@Time    :2019/9/2 15:18
#@Author  :XiaoMa
import re,math,random

def zero_digits(s):
    """replace every digit in a string by zero"""
    return re.sub('\d','0',s)

def create_dico(item_list):
    assert type(item_list) is list
    dico={}
    for items in item_list:
        for item in items:
            # if item not in dico:
            #     dico[item]=1
            # else:
            #     dico[item]+=1
            dico[item]=dico.get(item,0)+1
    return dico

def create_mapping(dico):
    """create a mapping(item to id /id to item) from a dictionary.
        items are orded by decreasing frequency
    """
    sorted_items=sorted(dico.items(),key=lambda x:(-x[1],x[0]))
    id_to_item={i:v[0] for i,v in enumerate(sorted_items)}
    item_to_id={v:i for i,v in id_to_item.items()}
    return item_to_id,id_to_item

class BatchManager(object):
    def __init__(self,data,batch_size):
        self.batch_data=self.sort_and_pad(data,batch_size)
        self.len_data=len(self.batch_data)
    def sort_and_pad(self,data,batch_size):
        num_batch=len(data)//batch_size
        sorted_data=sorted(data,key=lambda x:len(x[0]))
        batch_data=[]
        for i in range(num_batch):
            batch_data.append(self.arrange_batch(sorted_data[int(i*batch_size):int((i+1)*batch_size)]))
        return batch_data

    @staticmethod
    def arrange_batch(batch):
        """
        把batch整理成为一个[5, ]的数组
        :param batch:
        :return:
        """
        strings=[]
        segment_ids=[]
        chars=[]
        masks=[]
        targets=[]
        for string,seg_ids,char,mask,target in batch:
            strings.append(string)
            segment_ids.append(seg_ids)
            chars.append(char)
            masks.append(mask)
            targets.append(target)
        return [strings,segment_ids,chars,masks,targets]

    @staticmethod
    def pad_data(data):
        strings=[]
        chars=[]
        segs=[]
        targets=[]
        max_length=max([len(sentence[0]) for sentence in data])
        for line in data:
            string,segment_ids,char,seg,target=line
            padding=[0]*(max_length-len(string))
            strings.append(string+padding)
            chars.append(char+padding)
            segs.append(seg+padding)
            targets.append(target+padding)
        return [strings,chars,segs,targets]

    def iter_batch(self,shuffle=False):
        if shuffle:
            return random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
