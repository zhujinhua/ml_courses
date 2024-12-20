### 1.卷积模块
### 2.BatchNorm组件
    - Batch:批量
    - Norm: 规范化 深度学习典型的人工智能，由数据决定，从数据里面获取规则
        - (x - mu)/sigma
### 3.ReLU 层
### 4.深度学习人物
    - 三巨头 2006 Geoffey Hinton, Yoshua Bengio, Yann LeCun(CV, LeNet 1998, Facebook), 吴恩达 网红科普大神， 李飞飞 数据集大神 ImageNet， 何凯明 ResNet->transformer
    - LeCun LeNet-5 看一下 1998: 分为两步：Feature Extraction, Traniable Classifier,subsampling池化, 卷积：越卷越多层，池化图片越来越小 kernal 5*5
### 5. 复习
    - 论文
    - LetNet
    - 专业名称,各个层的含义，输入输出是什么
    - 图片分类应用

### 6.概念梳理
    6.1 卷积层（Convolutional Layer）
        *主要功能
            - 特征提取：卷积层通过应用卷积核（filter）提取输入图像中的局部特征，如边缘、纹理等。
            - 参数共享：卷积核在整个输入图像上滑动，减少了模型的参数数量，提高了计算效率。
            - 空间关系保持：卷积操作保留了输入图像的空间信息，即相邻像素的关系。
        *工作原理
            - 卷积操作：卷积核在输入图像上滑动，对每个位置执行元素级的乘法和加法操作，生成特征图。
            - 激活函数：通常在卷积操作后应用非线性激活函数（如ReLU）增加模型的非线性表达能力。
    6.2 池化层（Pooling Layer）
        *主要功能
            - 降维：池化层通过下采样操作减少特征图的尺寸，降低计算复杂度。
            - 特征保持：在降维的同时保留输入图像的重要特征。
            - 防止过拟合：通过减少特征图的维度，防止模型过度拟合训练数据。
        *工作原理
            - 最大池化（Max Pooling）：选择池化窗口中的最大值作为输出值。
            - 平均池化（Average Pooling）：选择池化窗口中的平均值作为输出值。
    假设我们有一个输入图像尺寸为 224x224，并经过一系列的卷积和池化操作：
    
    第一层卷积后，特征图可能变为 222x222（假设使用3x3的卷积核且无填充）。
    第一层池化后，特征图可能变为 111x111（假设使用2x2的最大池化，stride为2）。
    第二层卷积后，特征图可能变为 109x109（同样假设使用3x3的卷积核且无填充）。
    第二层池化后，特征图可能变为 54x54（假设再次使用2x2的最大池化，stride为2）。