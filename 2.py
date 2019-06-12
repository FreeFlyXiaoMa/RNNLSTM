m=[[1,2,3],[4,5,6],[7,8,9]]

n=[[1,1,1],[2,2,3],[3,3,3]]

p=[[1,1,1],[2,2,2]]

#print(list(zip(m,p)))
value=[x*y for a,b in zip(m,n) for x,y in zip(a,b)]
print(value)
