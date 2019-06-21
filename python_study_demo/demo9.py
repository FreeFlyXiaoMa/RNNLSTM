# -*- coding: utf-8 -*-
#@Time   :2019/6/20 18:30
#@Author :XiaoMa
#@email  :

import threading
import time
"""
threading.Thread()
"""
class MyThread(threading.Thread):
    def __iter__(self):
        threading.Thread.__init__(self)
    def run(self):
        global n,lock
        time.sleep(1)
        if lock.acquire():
            print(n,self.name)
            n+=1
            lock.release()

if __name__=="__main__":
    n=1
    ThreadList=[]
    lock=threading.Lock()
    for i in range(1,200):
        t=MyThread()
        ThreadList.append(t)
    for t in ThreadList:
        t.start()
    for t in ThreadList:
        t.join()

