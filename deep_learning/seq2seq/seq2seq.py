from torch import nn
from encoder import Encoder
from decoder import Decoder


class Seq2Seq(nn.Module):

    def __init__(self, tokenizer):
        super(Seq2Seq, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = Encoder(self.tokenizer)
        self.decoder = Decoder(self.tokenizer)

    def forward(self, x, x_len, y, y_len):
        """
            训练时的正向传播
        """
        out, hn = self.encoder(x, x_len)
        results = self.decoder(hn, y, y_len)
        # [seq_len, batch_size, dict_len]
        return results

    def batch_infer(self, x, x_len):
        """
            批量推理
        """
        out, hn = self.encoder(x, x_len)
        preds = self.decoder.batch_infer(hn)
        results = []
        for s in preds.t():
            results.append(self.tokenizer.decode_output(s.tolist()))
        return results
