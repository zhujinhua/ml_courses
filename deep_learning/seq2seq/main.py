import random
from tokenizer import get_tokenizer
from dataloader import get_dataloader
from seq2seq import Seq2Seq
import torch
from torch import nn
import os


class Translation(object):
    """
    把任务封装为一个类
    """

    def __init__(self, data_file="../../dataset/translate.txt"):
        self.data_file = data_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = get_tokenizer(data_file=data_file)
        self.model = self._get_model()
        self.epochs = 20
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.output_word2idx.get("<PAD>")
        )

    def _get_model(self):
        """
        获取模型
        """
        # 实例化模型
        model = Seq2Seq(tokenizer=self.tokenizer).to(device=self.device)
        # 加载权重
        if os.path.exists("./model.pt"):
            model.load_state_dict(state_dict=torch.load(f="./model.pt"))
            print("加载本地模型成功")
        return model

    def get_loss(self, decoder_outputs, y):
        decoder_outputs = decoder_outputs.to(device=self.device)
        # [batch_size]
        y = y.contiguous().view(-1)
        # [batch_size, dim]
        decoder_outputs = decoder_outputs.contiguous().view(
            -1, decoder_outputs.size(-1)
        )
        # 计算损失
        loss = self.loss_fn(decoder_outputs, y)
        return loss

    def get_real_output(self, y):
        """
        将预测结果转换为真实结果
        """
        y = y.t().tolist()
        results = []
        for s in y:
            results.append(
                [
                    self.tokenizer.output_idx2word.get(idx)
                    for idx in s
                    if idx
                    not in [
                        self.tokenizer.output_word2idx.get("<EOS>"),
                        self.tokenizer.output_word2idx.get("<PAD>"),
                    ]
                ]
            )
        return results

    def get_real_input(self, x):
        """
        将输入转换为字符
        """
        x = x.t().tolist()
        results = []
        for s in x:
            results.append(
                [
                    self.tokenizer.input_idx2word.get(idx)
                    for idx in s
                    if idx not in [self.tokenizer.input_word2idx.get("<PAD>")]
                ]
            )
        return results

    def train(self):
        """
        训练模型
        """
        # 训练集加载器
        train_dataloader = get_dataloader(
            tokenizer=self.tokenizer, data_file=self.data_file, part="train"
        )

        # 训练过程
        is_complete = False
        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, (x, x_len, y, y_len) in enumerate(train_dataloader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                results = self.model(x, x_len, y, y_len)
                loss = self.get_loss(decoder_outputs=results, y=y)

                # 简单判定一下，如果损失小于0.5，则训练提前完成
                if loss.item() < 0.3:
                    is_complete = True
                    print(f"训练提前完成, 本批次损失为：{loss.item()}")
                    break

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # 过程监控
                with torch.no_grad():
                    if batch_idx % 100 == 0:
                        print(
                            f"第 {epoch + 1} 轮 {batch_idx + 1} 批, 当前批次损失: {loss.item()}"
                        )
                        x_true = self.get_real_input(x)
                        y_pred = self.model.batch_infer(x, x_len)
                        y_true = self.get_real_output(y)
                        samples = random.sample(population=range(x.size(1)), k=2)
                        for idx in samples:
                            print("\t真实输入：", x_true[idx])
                            print("\t真实结果：", y_true[idx])
                            print("\t预测结果：", y_pred[idx])
                            print(
                                "\t----------------------------------------------------------"
                            )

            # 外层提前退出
            if is_complete:
                # print("训练提前完成")
                break
        # 保存模型
        torch.save(obj=self.model.state_dict(), f="./model.pt")

    def infer(self, x="Am I wrong?"):
        """
        单样本推理
        """
        print("输入：", x)
        # 分词
        x = self.tokenizer.split_input(x)
        print("分词：", x)
        # 编码
        x = self.tokenizer.encode_input(x, len(x))
        print("编码：", x)
        # 张量
        x = torch.tensor(data=[x], dtype=torch.long).t().to(device=self.device)
        print("张量：", x)
        # 长度
        x_len = torch.tensor(data=[len(x)], dtype=torch.long)
        # 评估模式
        self.model.eval()
        # 无梯度环境
        with torch.no_grad():
            y_pred = self.model.batch_infer(x, x_len)
            print("原始输出：", y_pred[0][0])
            print("最终输出：", y_pred[0][1])
            return y_pred[0]


if __name__ == "__main__":
    translation = Translation()
    # 模型训练
    translation.train()
    # 模型推理
    y_pred = translation.infer(x="Am I wrong?")
