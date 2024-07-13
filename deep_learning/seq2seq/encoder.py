import torch
from torch import nn


class Encoder(nn.Module):
    """
        定义一个 编码器
    """

    def __init__(self, tokenizer):
        super(Encoder, self).__init__()
        self.tokenizer = tokenizer
        # 嵌入层
        self.embed = nn.Embedding(num_embeddings=self.tokenizer.input_dict_len,
                                  embedding_dim=self.tokenizer.input_embed_dim,
                                  padding_idx=self.tokenizer.input_word2idx.get("<PAD>"))
        # GRU单元
        self.gru = nn.GRU(input_size=self.tokenizer.input_embed_dim,
                          hidden_size=self.tokenizer.input_hidden_size,
                          batch_first=False)

    def forward(self, x, x_len):
        # [seq_len, batch_size] --> [seq_len, batch_size, embed_dim]
        x = self.embed(x)
        # 压紧被填充的序列
        x = nn.utils.rnn.pack_padded_sequence(input=x,
                                              lengths=x_len,
                                              batch_first=False)
        out, hn = self.gru(x)
        # 填充被压紧的序列
        out, out_len = nn.utils.rnn.pad_packed_sequence(sequence=out,
                                                        batch_first=False,
                                                        padding_value=self.tokenizer.input_word2idx.get("<PAD>"))
        # out: [seq_len, batch_size, hidden_size]
        # hn: [1, batch_size, hidden_size]
        return out, hn


if __name__ == '__main__':
    from tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    encoder = Encoder(tokenizer)
    print(encoder)
    x = torch.randint(low=0, high=tokenizer.input_dict_len, size=(16, 2), dtype=torch.long)
    x_len = torch.tensor(data=[16, 16], dtype=torch.long)
    out, hn = encoder(x, x_len)
    print(out.shape)
    print(hn.shape)