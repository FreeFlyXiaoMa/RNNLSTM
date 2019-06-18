class A(object):
    name='python'#类属性
    def __init__(self):
        self.age=18 #实例属性
    def a_print(self):#实例方法
        print('aaa')

    @classmethod    #类方法
    def b_print(cls):   #cls--指代当前的类
        print(cls.name)

    @staticmethod   #静态方法
    def c_print():
        print('static method')

print(A.name)   #类调用属性
#A.a_print()
A.b_print()     #类调用类方法
A.c_print()     #类调用静态方法

a=A()   #生成类实例
a.a_print()     #实例对象调用实例方法
a.b_print()     #调用类方法
a.c_print()     #调用静态方法

class Parent:
    def __init__(self):
        print('this is a parent init.')
    def sleep(self):
        print('parent sleeps')

class Child(Parent):
    def __iter__(self):
        print('this is a child init222.')
    def __init__(self):
        super().__init__()
        print('this is a child init.')
    def sleep(self):
        print('child sleeps')
c=Child()
c.__iter__()
c.sleep()



