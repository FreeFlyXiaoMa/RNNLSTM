"""
线程
"""
from time import *
from datetime import datetime
tickets=[
    ['2018-4-7 8:00','北京','沈阳',10,120],
    ['2018-4-7 9:00','上海','宁波',5,100],
    ['2018-4-7 12:00','天津','北京',20,55],
    ['2018-4-7 14:00','广州','武汉',0,200],
    ['2018-4-7 16:00','重庆','西安',3,180],
    ['2018-4-7 18:00','深圳','上海',49,780],
    ['2018-4-7 18:10','武汉','长沙',10,210]
]

def buy_ticket(name,nums,data1,start_station):
    i=0
    sleep(1)
    for get_record in tickets:
        if data1==get_record[0] and start_station ==get_record[1]:
            if get_record[3] >= nums:
                tickets[i][3]=get_record[3]-nums
                return nums
            else:
                print('%s现存票数量不够，无法购买！'%(name))
                return -1
            i+=1
        print('%s今日无票，无法购买'%(name))

if __name__=='__main__':
    print('开始时间：',datetime.now())
    result=buy_ticket('张山',3,'2018-4-7 9:00','上海')
    if result>0:
        print('张山购买%d张票成功！'%(3))
    result=buy_ticket('李四',1,'2018-4-7 14:00',"广州")
    if result >0:
        print('李四购买%d张票成功！'%(1))
    result=buy_ticket('王五',2,'2018-4-7 9:00','上海')
    if result>0:
        print('王五购买%d张票成功！'%(2))
    print('剩余票数为：\n')
    for ticket in tickets:
        print(ticket)






















