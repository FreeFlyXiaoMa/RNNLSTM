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

from multiprocessing import Process,Pipe
import multiprocessing
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

def do_send(conn_s,j):
    dic={
        '发送序号':j,
        '鲫鱼':[18,10.5],
        '鲤鱼':[8,7,2]
    }
    conn_s.send(dic)
    conn_s.close()

if __name__=='__main__':
    """for data in range(do_thread_num*2):
        q_data.put(data)
        for j in range(do_thread_num):
            t1=MyThread(getOne,q_data,j).start()
    pool=multiprocessing.Pool()
    print('开始时间：',datetime.now())

    for i in range(10):
        pool.apply_async(do1,(i,))
        pool.close()
        pool.join()
    print('结束时间：',datetime.now())"""

    receive_conn,send_conn=Pipe()
    i=0
    while i<2:
        i+=1
        pp=Process(target=do_send,args=(send_conn,i))
        pp.start()
        print('接收数据%s成功！'%(receive_conn.recv()))
        pp.join()