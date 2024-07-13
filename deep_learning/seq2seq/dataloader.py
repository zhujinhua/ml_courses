import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os


class Seq2SeqDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, data_file, tokenizer, part="train"):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.part = part
        self.data = None
        self._load_data()

    def _load_data(self):
        if os.path.exists(f"./{self.part}.bin"):
            self.data = torch.load(f=f"./{self.part}.bin")
            print("加载本地数据集成功")
            return

        data = []
        with open(file=self.data_file, mode="r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                if line:
                    input_sentence, output_sentence = line.strip().split("\t")
                    input_sentence = self.tokenizer.split_input(input_sentence)
                    output_sentence = self.tokenizer.split_output(output_sentence)
                    data.append([input_sentence, output_sentence])
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)
        if self.part == "train":
            self.data = train_data
        else:
            self.data = test_data

        # 保存数据
        torch.save(obj=self.data, f=f"./{self.part}.bin")

    def __getitem__(self, idx):
        """
        返回一个样本
            - 列表格式
            - 内容 + 实际长度
        """

        input_sentence, output_sentence = self.data[idx]
        return (
            input_sentence,
            len(input_sentence),
            output_sentence,
            len(output_sentence),
        )

    def __len__(self):
        return len(self.data)


def collate_fn(batch, tokenizer):
    # 根据 x 的长度来 倒序排列
    batch = sorted(batch, key=lambda ele: ele[1], reverse=True)
    # 合并整个批量的每一部分
    input_sentences, input_sentence_lens, output_sentences, output_sentence_lens = zip(
        *batch
    )

    # 转索引【按本批量最大长度来填充】
    input_sentence_len = input_sentence_lens[0]
    input_idxes = []
    for input_sentence in input_sentences:
        input_idxes.append(tokenizer.encode_input(input_sentence, input_sentence_len))

    # 转索引【按本批量最大长度来填充】
    output_sentence_len = max(output_sentence_lens)
    output_idxes = []
    for output_sentence in output_sentences:
        output_idxes.append(
            tokenizer.encode_output(output_sentence, output_sentence_len)
        )

    # 转张量 [seq_len, batch_size]
    input_idxes = torch.LongTensor(input_idxes).t()
    output_idxes = torch.LongTensor(output_idxes).t()
    input_sentence_lens = torch.LongTensor(input_sentence_lens)
    output_sentence_lens = torch.LongTensor(output_sentence_lens)

    return input_idxes, input_sentence_lens, output_idxes, output_sentence_lens


def get_dataloader(tokenizer, data_file="../../dataset/translate.txt", part="train", batch_size=32):
    dataset = Seq2SeqDataset(data_file=data_file, tokenizer=tokenizer, part=part)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if part == "train" else False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    return dataloader


if __name__ == "__main__":
    from tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    dataloader = get_dataloader(tokenizer=tokenizer)
    for batch in dataloader:
        print(batch)
        break
