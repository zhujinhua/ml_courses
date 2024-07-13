import torch
from torch import nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(self, tokenizer):
        super(Decoder, self).__init__()
        self.tokenizer = tokenizer
        # 嵌入
        self.embed = nn.Embedding(
            num_embeddings=self.tokenizer.output_dict_len,
            embedding_dim=self.tokenizer.output_embed_dim,
            padding_idx=self.tokenizer.output_word2idx.get("<PAD>"),
        )
        # 抽取特征
        self.gru = nn.GRU(
            input_size=self.tokenizer.output_embed_dim,
            hidden_size=self.tokenizer.output_hidden_size,
            batch_first=False,
        )
        # 转换维度，做概率输出
        self.fc = nn.Linear(
            in_features=self.tokenizer.output_hidden_size,
            out_features=self.tokenizer.output_dict_len,
        )

    def forward_step(self, decoder_input, decoder_hidden):
        """
        单步解码:
            decoder_input: [1, batch_size]
            decoder_hidden: [1, batch_size, hidden_size]
        """
        # [1, batch_size] --> [1, batch_size, embedding_dim]
        decoder_input = self.embed(decoder_input)
        # 输入：[1, batch_size, embedding_dim] [1, batch_size, hidden_size]
        # 输出：[1, batch_size, hidden_size] [1, batch_size, hidden_size]
        # 因为只有1步，所以 out 跟 decoder_hidden是一样的
        out, decoder_hidden = self.gru(decoder_input, decoder_hidden)
        # [batch_size, hidden_size]
        out = out.squeeze(dim=0)
        # [batch_size, dict_len]
        out = self.fc(out)
        # out: [batch_size, dict_len]
        # decoder_hidden: [1, batch_size, hidden_size]
        return out, decoder_hidden

    def forward(self, encoder_hidden, y, y_len):
        """
        训练时的正向传播
            - encoder_hidden: [1, batch_size, hidden_size]
            - y: [seq_len, batch_size]
            - y_len: [batch_size]
        """
        # 计算输出的最大长度（本批数据的最大长度）
        output_max_len = max(y_len.tolist()) + 1
        # 本批数据的批量大小
        batch_size = encoder_hidden.size(1)
        # 输入信号 SOS  读取第0步，启动信号
        # decoder_input: [1, batch_size]
        # 输入信号 SOS [1, batch_size]
        decoder_input = torch.LongTensor(
            [[self.tokenizer.output_word2idx.get("<SOS>")] * batch_size]
        ).to(device=device)
        # 收集所有的预测结果
        # decoder_outputs: [seq_len, batch_size, dict_len]
        decoder_outputs = torch.zeros(
            output_max_len, batch_size, self.tokenizer.output_dict_len
        )
        # 隐藏状态 [1, batch_size, hidden_size]
        decoder_hidden = encoder_hidden
        # 手动循环
        for t in range(output_max_len):
            # 输入：decoder_input: [batch_size, dict_len], decoder_hidden: [1, batch_size, hidden_size]
            # 返回值：decoder_output_t: [batch_size, dict_len], decoder_hidden: [1, batch_size, hidden_size]
            decoder_output_t, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            # 填充结果张量 [seq_len, batch_size, dict_len]
            decoder_outputs[t, :, :] = decoder_output_t
            # teacher forcing 教师强迫机制
            use_teacher_forcing = random.random() > 0.5
            # 0.5 概率 实行教师强迫
            if use_teacher_forcing:
                # [1, batch_size] 取标签中的下一个词
                decoder_input = y[t, :].unsqueeze(0)
            else:
                # 取出上一步的推理结果 [1, batch_size]
                decoder_input = decoder_output_t.argmax(dim=-1).unsqueeze(0)
        # decoder_outputs: [seq_len, batch_size, dict_len]
        return decoder_outputs

    def batch_infer(self, encoder_hidden):
        """
        推理时的正向传播
            - encoder_hidden: [1, batch_size, hidden_size]
        """
        # 推理时，设定一个最大的固定长度
        output_max_len = self.tokenizer.output_max_len
        # 获取批量大小
        batch_size = encoder_hidden.size(1)
        # 输入信号 SOS [1, batch_size]
        decoder_input = torch.LongTensor(
            [[self.tokenizer.output_word2idx.get("<SOS>")] * batch_size]
        ).to(device=device)
        # print(decoder_input)
        results = []
        # 隐藏状态
        # encoder_hidden: [1, batch_size, hidden_size]
        decoder_hidden = encoder_hidden
        with torch.no_grad():
            # 手动循环
            for t in range(output_max_len):
                # decoder_input: [1, batch_size]
                # decoder_hidden: [1, batch_size, hidden_size]
                decoder_output_t, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden
                )
                # 取出结果 [1, batch_size]
                decoder_input = decoder_output_t.argmax(dim=-1).unsqueeze(0)
                results.append(decoder_input)
            # [seq_len, batch_size]
            results = torch.cat(tensors=results, dim=0)
        return results
