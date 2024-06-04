# 机器学习解决NLP问题
import pandas as pd
import jieba # 按短语分词, 去除停顿词
df = pd.read_csv('中文外卖评论数据集.csv')
print(df[:, 1])