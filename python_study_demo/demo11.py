# -*- coding: utf-8 -*-
#@Time    :2019/6/22 11:36
#@Author  :XiaoMa
#@email   :

import queue
import threading
import time
import random
q_data=queue.Queue(10)  #10个元素的实例
do_thread_num=5

def getOne(one,j):
    time.sleep(random.random()*3)   #随机睡眠0-3秒
    #print('线程序号%d，获取元素%d'%(j,one))

class MyThread(threading.Thread):
    def __init__(self,func,data,j):
        threading.Thread.__init__(self)
        self.func=func
        self.data=data
        self.j=j
    def run(self):
        while self.data.qsize()>0:
            self.func(self.data.get(),self.j)

from multiprocessing import Process
from datetime import datetime
def do1(j):
    print('第%d个进程！'%(j))

class MyProcess(Process):
    def __init__(self,target,args):
        Process.__init__(self)
        self.target=target
        self.args=args
    def run(self):
        self.target(self.args)


if __name__=='__main__':
    for data in range(do_thread_num*2):
        q_data.put(data)
        for j in range(do_thread_num):
            t1=MyThread(getOne,q_data,j).start()

    print('开始时间：',datetime.now())
    for i in range(10):
        p1=MyProcess(target=do1,args=(i,))
        p1.start()
        p1.join()

    print('结束时间：',datetime.now())