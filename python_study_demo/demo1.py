"""
自定义函数
"""
def find_factors(nums):
    if type(nums) != int:
        print('输入值类型出错')
    if nums <=0:
        print('输入值范围出错')

    i=1
    str1=''
    while i <=nums:
        if nums % i==0:
            str1+=' '+str(i)
        i+=1
    return str1
#print(help(find_factors))
import  sys
sys.path[0]='/pyWorkspace/'

print(find_factors(20))


































