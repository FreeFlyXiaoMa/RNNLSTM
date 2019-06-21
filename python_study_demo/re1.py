import re

tt ='Tina is a good girl,she is cool,clever, and so on...'

patt=re.compile(r'\w*oo\w*')
print(re.findall(patt,tt))

print(re.match('com','comwwwww.runconoob').group())
print(re.match('com','CoMEEEE',re.I).group())

print(re.search('\dcom','ddd2dcom.333ss.1com').group())
p=re.compile(r'\d+')
print(re.findall(p,'o1m2n3b4'))

tt ='Tina is a good girl,she is cool,clever, and so on...'
print(re.findall('(\w*)oo(\w)',tt))

print(re.split(r'\d+','ddd1eeeee23ddddd45mmmm6'))

tt ='Tina is a good girl,she is cool,clever, and so on...'
print(re.subn('\s+','-',tt))


a=re.findall(r'a(\d+?)','a23456dddd')
print(a)
b=re.findall(r'a(\d+)','a23456drr456dddd')
print(b)



