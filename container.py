import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_word_table, import_vecs, import_corpus

# 词表
tblt = get_word_table()
# 有限长的index -> vecs
w2v_vecs = import_vecs('wiki_word2vec_50.bin', tblt)


# 按照示例中给予的TextCNN图来创建CNN
class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = "TextCNN"
        class_num = 2
        vocab_size = len(tblt)
        update_w2v = True
        embedding_dim = 50
        kernel_num = 30
        kernel_size = [2, 3, 5, 7]
        drop_prob = 0.3
        pretrained_embed = torch.Tensor(w2v_vecs)

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(pretrained_embed)

        # conv layer
        self.convs = nn.ModuleList([
            nn.Conv2d(1, kernel_num, (ks, embedding_dim)) for ks in kernel_size
        ])

        # dropout
        self.dropout = nn.Dropout(drop_prob)

        # all_to_all layer
        self.fc = nn.Linear(in_features=len(kernel_size) * kernel_num, out_features=class_num)

    @staticmethod
    def patching(inputx, conv):
        inputx = F.relu(conv(inputx).squeeze(3))
        return F.max_pool1d(inputx, inputx.shape[2]).squeeze(2)

    def forward(self, inputx):
        inputx = self.embedding(inputx.to(torch.int64)).unsqueeze(1)
        patches = [self.patching(inputx, conv) for conv in self.convs]
        return F.log_softmax(self.fc(self.dropout(torch.cat(patches, 1))), dim=1)


class LstmRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_num = 2
        self.layers = 4
        vocab_size = len(tblt)
        update_w2v = True
        embedding_dim = 50
        drop_prob = 0.3
        pretrained_embed = torch.Tensor(w2v_vecs)
        self.__name__ = "TextRNN"

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(pretrained_embed)

        # code layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=100,
            num_layers=self.layers,
            bidirectional=True,
            dropout=drop_prob
        )
        self.parse = nn.Linear(100, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, self.class_num)

    def forward(self, inputx):
        embedded = self.embedding(inputx.to(torch.int64)).permute(1, 0, 2)
        output_s, (hidden_last, cell) = self.lstm(embedded)
        res = self.fc2(self.fc1(self.parse(hidden_last[-1])))
        return res


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.class_num = 2
        vocab_size = len(tblt)
        update_w2v = True
        embedding_dim = 50
        pretrained_embed = torch.Tensor(w2v_vecs)
        self.__name__ = "TextMLP"

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad = update_w2v
        self.embedding.weight.data.copy_(pretrained_embed)

        # all_to_all layer
        self.fc1 = nn.Linear(embedding_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, self.class_num)

    def forward(self, inputx):
        embedded = self.embedding(inputx.to(torch.int64))
        res = self.fc1(embedded)
        res = self.relu(res)
        res = res.permute(0, 2, 1)  # shape = [length, sentence_len, hidden] => [length, hidden, class_num]
        res = F.max_pool1d(res, res.shape[-1]).squeeze(-1)
        # shape = [length, hidden, class_num] => [length, hidden]
        res = self.fc2(res)
        return res


if __name__ == '__main__':
    model = MLP()
    data, accs = import_corpus("test.txt", tblt)
    data = torch.from_numpy(data)
    x = model.embedding(data.to(torch.int64))
    print(x.shape)
    x = model.relu(model.fc1(x))
    print(x.shape)
    x = F.max_pool1d(x, x.shape[-1]).squeeze(-1)
    print(x.shape)
