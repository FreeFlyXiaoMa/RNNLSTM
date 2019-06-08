import numpy as np
import pandas as pd
#print(np.random.rand(2,3))

class_data={'Names':['John','Ryan','Emily'],
            'Standard':[7,5,8],
            'Subject':['English','Mathematics','Science']}

class_df=pd.DataFrame(class_data,index=['student1','student2','student3'],columns=['Names','Standard','Subject'])

print(class_df)
class_df.ix['student4']=['Robin',np.nan,'History']
#print(class_df.T)
print(class_df.sort_values('Standard'))

clo_entry=pd.Series(['A','B','A+','C'],index=['student1','student2','student3','student4'])

class_df['Grade']=clo_entry
print(class_df)

class_df.fillna(10,inplace=True)
print(class_df)

student_age=pd.DataFrame(data={'age':[13,10,15,18]},index=['student1','student2','student3','student4'])
print(student_age)
class_data=pd.concat([class_df,student_age],axis=1)

print(class_data)

class_data['Subject']=class_data['Subject'].map(lambda x:x+'Sub')
print(class_data['Subject'])

def age_add(x):
    return x+1

print('-----old ages-----')
print(class_data['age'])
print('----new values----')
print(class_data['age'].apply(age_add))

print(class_data.Grade.dtypes)
class_data['Grade']=class_data['Grade'].astype('category')
print(class_data.Grade.dtypes)
import nltk