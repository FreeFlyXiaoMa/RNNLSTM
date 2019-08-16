# -*- coding: utf-8 -*-
#@Time    :2019/8/16 14:59
#@Author  :XiaoMa
import threading
def ontime_one():
    for i in range(10):
        print('?')

def ontime_two():
    for i in range(10):
        print('!')
threads=[]
t1=threading.Thread(target=ontime_one)
threads.append(t1)
t2=threading.Thread(target=ontime_two)
threads.append(t2)

for t in threads:
    t.setDaemon(True)   #守护线程，线程空闲后，自动回收线程
    t.start()
