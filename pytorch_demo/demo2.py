# -*- coding: utf-8 -*-
#@Time    :2019/6/26 21:01
#@Author  :XiaoMa

import torch as t
import numpy as np
import logging
import logging.handlers


from torch.autograd import Variable



# def get_logger():
#     LOG_FILE = r'D:\\Documents\book_tool.log'
#
#     handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5, encoding='utf-8')
#     fmt = '%(asctime)s - %(levelname)s - %(message)s'
#
#     formatter = logging.Formatter(fmt)
#     handler.setFormatter(formatter)
#
#     logger = logging.getLogger('book_tool')
#     logger.addHandler(handler)
#     logger.setLevel(logging.DEBUG)
#     return logger
#
#
# import os
# logger=get_logger()
# logging.getLogger().setLevel(logging.DEBUG)
# logging.basicConfig(filename=os.path.join(os.getcwd(),'log.txt'),level=logging.DEBUG)
#
# x=Variable(t.ones(2,2),requires_grad=True)
# print('111111')
#
# print(x)

import sys
import time
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a",encoding='utf-8')     #文件权限为'a'，追加模式

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("D:\\pyWorkspace\pytorch_demo\log.txt")  # 保存到D盘
print('*'*20,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'*'*20)
print('输出结果为：..................................')

