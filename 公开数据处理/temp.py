import pandas as pd


data = pd.read_csv('D:\python_common\代码_感知质量大论文\公开数据处理\MAMS\关键词抽取结果_train.csv')['review']

l = 0
for i in data:
    l += len(i.split(' '))
print(l/len(data))
