# -*- coding: utf-8 -*-
#@Time    :2019/7/18 11:55
#@Author  :XiaoMa
import pandas as pd
df = pd.DataFrame({'key1':list('aabba'),
                  'key2': ['one','two','one','two','one'],
                  'data1': ['1','3','5','7','9'],
                  'data2': ['2','4','6','8','10']})

print(df.groupby(['key1']).size().reset_index())
