def r_dichotomy(nums,find,left,right):
    middle=(left+right)//2
    if nums[middle] == find:
        return middle

    if nums[middle] <find:
        left=middle
        return r_dichotomy(nums,find,left,right)
    elif nums[middle] > find:
        right=middle
        return r_dichotomy(nums,find,left,right)

nums=[1,2,3,4,5,6,7,8,9]
print(r_dichotomy(nums,3,0,len(nums)-1))