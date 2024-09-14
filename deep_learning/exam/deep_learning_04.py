"""
    4，对指定语料（corpus文件夹下所有内容）构建字典（可借助 jieba 分词）；
"""
import jieba
import os

# 读取语料
root = r".\corpus"
words = set()
for file in os.listdir(root):
    file_path = os.path.join(root, file)
    with open(file=file_path, mode="r", encoding="utf8") as f:
        data = jieba.lcut(f.read())
        words = words.union(set(data))

# 构建字典
word2idx = {word: idx for idx, word in enumerate(words)}
idx2word = {idx: word for word, idx in word2idx.items()}
