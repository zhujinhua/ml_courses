### 图像分割
    - 语义分割
        - class 级别分割（比如分割鹦鹉，只要是鹦鹉就识别分割）
        - 实现策略：
            - 像素级图像分类：把每个像素都做一个N+1分类
        - 输入：一张图像
        - 输出：mask图像，蒙版图像
        - 算法原理：
            - 编解码器原理：EncoderDecoder模型
            - Encoder:把一个具体的实体Entity编码为一个抽象的中间表达（vector），把原始数据进一步做数字化处理，变成一个语义化的表达
            - Decoder:根据编码器得到的中间表达，解码出相应的结果
            - context vector 中间表达：链接了decoder,decoder模型
            - U-Net：Conv networks for bimedicine image segment
                - Encoder, Decoder思想，内化，泛化，类似模拟大脑学习，比如不断抽取学习什么事肝脏，以小博大的过程
            - 多尺度特征 multi-scale特征
                - 信息的抽象层次不同的特征
                - 多尺度特征融合：把不同scale的特征拼接concatenate!!!!!
    - 实例分割
        - Object 级别分割
        - 目标检测+语义分割（像素级分类） 同时进行（yolact merge into yolo）
    - 全景分割
        - 把整个图像都做分割，包括背景在内
    - Segment Anything Model(SAM):交互，模型比较大，魔塔上

### 数据集
    - ImageNet 图像分类：1000
    - coco
    - voc
### 实例分割数据标注
    - 1.使用labelme标注每一个物体：如果存在遮挡，则需要分块标注
    - 2.把标注结果转化为类似coco的格式
    - 3.利用YOLO提供的多段合成算法合并多个分段