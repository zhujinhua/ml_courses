# ml_courses
1. traditional machine learning algorithm
   - 算法角度
      - 分类算法：逻辑回归，KNN，贝叶斯，决策树，支持向量机，集成学习
      - 回归算法：线性回归，KNN，决策树，支持向量机，集成学习
      - 降维算法：PCA, SVD
      - 聚类算法：K-Means
   - 数据角度
       - 特征：互相独立independently，表格类数据 tabular data 
   - 项目流程
       - 分析项目，确定输入，输出
       - 根据输入，输出构建数据集（爱数科数据）
       - 遴选算法，完成输入到输出的映射
       - 模型部署，上线推理
   - 项目案例：
     - 外卖评论：情感识别（positive, negative, neutral）
     - 薪资区间预测
   - NLP问题
     - 不能直接处理文字，如何把文本向量化：举例
       - S1：你吃了吗？
       - S2：我吃了！
       - S3：下班了吗？
     - 特征不是成行成列的规范结构
     - 第一步分词(按字来拆，语义单元，词元), 你 吃 了 吗 ？
     - 第二步：构建字典，去重(你，吃，了， 吗，？，!, 我，下，班)
     - 第三步：向量化句子 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
       - onehot:好处各个状态之间默认无关，坏处：稀疏向量
     - 特点：
       - 行：每个句子的长度都是相同的，都是字典长度！
       - 列：每个词在句子中出现的次数
2. deep learning 
3. 