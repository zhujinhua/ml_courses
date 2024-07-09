### OCR问题（光学字符识别） 提前图像中的文字信息
    - 行级语义，使用两阶段算法
    - 第一阶段：检测出行级文本区域
    - 第二阶段：识别这行的文字
    - PaddleOCR 模型：https://github.com/PaddlePaddle/PaddleOCR
### GAN生成对抗网络：主要用于模仿，生成图像（生成式模型的萌芽）2015年
    - G：生成器（造假）
    - A：鉴别器（打假）
    - 两个网络开始都没有任何能力，在竞争中共同发展，好比警察 VS 小偷
    - 收敛有时候比较困难
    - 最后把生成器拿出来，生成假样本
### Diffusion Model
    - 加噪，减噪
### Stable Diffusion Model扩散模型
    - 插入prompts
    - 有许多开源软件，比如魔塔（注册）：https://www.modelscope.cn/models/ming999/stable-diffusion-3-medium
### Word Embedding
    - [N, Seq_len, Embedding_dim]
### RNN:在大模型时代被淘汰了
    - 06年RNN引领NLP 
    - Understanding LSTM Networks：https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    