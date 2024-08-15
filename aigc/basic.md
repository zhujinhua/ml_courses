### 人工智能
    - 机器学习（统计机器学习，表格类数据）
    - 深度学习（计算机视觉 图像类数据 自然语言处理 时序数据）
    - 大模型（NLP大模型，CV大模型，多模态大模型（MM大模型））
### 大模型
    - "大"到哪里
    - 层数：几十层，几百层，几千层的transformer的encoder或者decoder
    - 参数层：B，billion,十亿，0.5B，6B，14B， 72B， 1000B...
        - 原理都是一样的
        - 参数量 + 数据量-->能力不一样（效果更好！！！）
    - 硬件要求：
        - NVIDIA独立显卡
        - 举例：6B 推理2/3倍的显存大小 12G-18G显存，训练：5倍，可以用lora方式训练，相对少
        - modelscope.cn
    - 数据量
        -T（万亿个，数量）
            - 在第一个阶段（与训练阶段PT），累计训练的数据量（token个数）
            - 15T
    - 训练时间
            - 3-6个月
### 大模型 Large Language Model/Big model
### 生成式人工智能，人工智能生成式内容：AIGC
    - 文生文
    - 文生图
    - 文生视频
    - 文生一切
### 人工智能
    - 判别式:P（y|x）
    - 生成式:P(x,y) creative
### 公共底层架构
    - transformer
    - 转换器
    - 目标：取代RNN，做序列转换
    - RNN存在的问题
        - 依赖循环，当序列很长时，无法并行计算
        - 通过隐藏状态按时序不断传递来实现时序信息的抽取，容易遗忘前面的信息
        - 通过层的堆叠无法获得足够的性能回报
    - Transformer优势：
        - 干掉循环
        - 通过自注意力来提取特征，不区分顺序；并行提取
        - 通过层的堆叠，可以获取足够的性能回报
    - attention源于CV，在NLP应用上火爆：注意力凸显，无关的信息乘以系数比如0.001，关注的信息乘以0.99
        - 初始，有很多特征可以使用
        - 注意力就是重点关注的有用的信息，削弱无用的信息
        - 实现方法：各个特征加权处理：
            - 重要的特征：权重比较大，突出重要性
            - 不重要的特征：权重比较小，削弱重要性
        - 外挂式注意力
        - 自注意力：根据自己所在的上下文求注意力！
### 大模型的三大架构
    - 直接利用 Transformer即可，使用Encoder-Decoder架构: T5（Text-to-Text Transfer Transformer）, Bart, 成为谷歌派
        - Transformer本身就是完整的生成式算法，所以，直接利用Transformer构建大模型，顺理成章
        - 优势：不用重新设计结构
        - 劣势：网络结构略显复杂：网络略显复杂，但是训练和推理逻辑简单
    - 只使用Transformer的Encoder，Encoder-Only架构（只使用一半，完成原来的全部功能）
        - 网络结构变简单，但训练和推理流程变复杂了
        - ChatGLM系列 质谱独有架构！！不被圈子认可，独一份
    - 只使用Transformer的Decoder, Decoder-Only架构（只使用一半，完成原来的全部功能）
        - OpenAI系列，被全世界认可，追捧

### 大模型的训练流程
    - 模型预训练 pre-train PT（最重要）
        - 预训练是内功修炼，不针对任何的任务！（比如用了15个T的语料）
        - 挖空填空（mask掩码预测预训练），完形填空；"打通任督二脉"，具有强大的文本理解能力
        - base版大模型，底座大模型，不能直接使用
    - 监督微调 Supervised Fine-Tuning SFT
        - 问答对，对话，知识编辑，知识注入（专家级）；chat版大模型（一般发布大模型两个版本：Base版大模型，Chat版大模型）
    - 基于人类反馈的强化学习 Reinforcement Learning with Human Feedback RLHF
        - ChatGPT 3的叫法，现在叫偏好对齐，纠正偏好

### 小模型的训练：
    - 步骤
        - 问题或场景的分析，搞定输入，输出
        - 构建数据集
        - 搭建、遴选模型
        - 训练
        - 评测
        - 部署
    - 特点：针对具体问题，采集具体数据，训练具体模型，使用具体模型；单一功能原则，单独维护原则，一个大的系统可以按需挂载看多小的模型
### 1. 普通模型的训练
- 分析需求（确定输入和输出）
- 构建数据集
- 搭建模型
- 训练模型
- 模型评估
- 模型部署

- 特点：
    - 单一功能
    - 定向训练
    - 定向维护

### 2. 大模型的训练

- Stage1：预训练 pretrain  PT
    - 修炼内功
    - 不针对任何特定的任务！！！
    - 耗时最长
    - 训练数据最多的
    - 不需要人工标注
    - 自监督（上自习）

- Stage2：监督微调 Supervised Fine-tuning SFT
    - 注入行业知识
    - 对齐：人类对话的能力
    - 问答对
    - 训练数据相对少一些
    - 需要专家编辑知识

- Stage3：偏好对齐 Preference Alignment 
- RLHF Reinforcement Learning with Human Feedback
    - 训练数据相对更少一些
    - 根据大模型的输出来构建相关数据集


### 3. BERT 训练

- BERT-Base: L = 12, H = 768, A = 12, Total parameters = 110M
- BERT-Large: L = 24, H = 1024, A = 16, Total parameters = 340M

### 训练前提：
  - 模型
  - 训练框架
### 训练框架
  - LLAMA-Factory
    - 把底层训练框架/技术套壳,通过壳，可以无代码进行模型训练
    - Steps: Refer to https://blog.csdn.net/weixin_37897145/article/details/138834235
      - pip install -e .[metrics] 
      - llamafactory-cli webui
### 训练一个医疗大模型
  ### 预训练

### 高性能部署
  - 作用：
    - 把大模型部署到服务端，暴露OpenAI Compatible API
    - 高性能，标准化，工业应用最广泛，最简单
  - 安装vLLM：pip install vllm
  - 部署方法： python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.3 --host 127.0.0.1 --port 8000
  - 调用方法：
    - langchain-openai
    - ChanOpenAI - 类似于Chat模型
    - OpenAI - 类似于Base模型
### 如何使用大模型
    - 第一个层面：外行作为工具来使用大模型
        - 直接使用各种产品即可
        - 关键词/文字提问，结果输出
        - 常见大模型：
    - 工业级应用
        - 大模型二次开发/上层/应用开发
        - 通过一些途径/方法，来释放大模型的强大NLP的能力
        - AI能力中台
        - 最根本
            - 训练
                - system:系统指令
                - user:用户指令和用户输入
                - tool: 工具描述
        - 使用
            prompt 提问，不同的提问方法（指令），会引出大模型不同的能力，让大模型做不同的事情！！！
        