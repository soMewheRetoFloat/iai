import numpy as np
import torch
import os, sys
import json
from tqdm import tqdm
import re
from gensim.models import KeyedVectors
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

DataPath = ".\Dataset"
chinese_char_pattern = re.compile(r'^[\u4E00-\u9FFF]+$')


# 判断中文字
def is_chinese(words) -> bool:
    return chinese_char_pattern.search(words) is not None


# vec_len = 50

def get_word_table():
    """
    获取词表
    :return: 一个词 -> idx字典，记录所有在原文中出现的词表的内容
    """
    word_table = Counter()
    ls = os.listdir(DataPath)
    for filename in ls:
        if filename.endswith(".txt") and filename.startswith("train"):
            with open(os.path.join(DataPath, filename), "r", encoding='utf-8') as f:
                for line in f.readlines():
                    sentence = line.strip().split()
                    #  第 0 位是label
                    for word in sentence[1:]:
                        if word not in word_table:
                            word_table[word] = len(word_table)
    return word_table


def import_vecs(path, word_table):
    """
    引入词向量
    :param word_table: 词表
    :param path:词向量文件名
    :return: 一个词 -> idx字典，一个idx -> 向量Tensor
    """
    vec_path = os.path.join(DataPath, path)
    init_model = KeyedVectors.load_word2vec_format(vec_path, binary=True)
    ret = np.zeros(shape=(len(word_table), 50), dtype=np.float32)
    for word in word_table.keys():
        try:
            ret[word_table[word]] = init_model[word]
        except KeyError:
            pass
    return ret


def import_corpus(path, word_table, sentence_max_length=120):
    """
    处理语料库文件
    :param sentence_max_length: 可容许的句子最大长度
    :param word_table：词表
    :param path: 语料库名
    :return: 语料库内容与标签
    """
    corpus_path = os.path.join(DataPath, path)
    ctts = np.zeros(sentence_max_length, dtype=np.float32)
    labels = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc="Importing corpus from " + corpus_path):
            sentence = line.strip().split()
            # 取出标签
            labels.append(int(sentence[0]))

            sequence = np.array([word_table.get(w, 0) for w in sentence[1:]])[:sentence_max_length]
            pad = max(0, sentence_max_length - len(sequence))
            padded = np.pad(sequence, (0, pad))
            ctts = np.vstack((ctts, padded))
    # ctts = np.transpose(ctts, (1, 0))
    ctts = np.delete(ctts, 0, axis=0)
    labels = np.array(labels)
    return ctts, labels

if __name__ == '__main__':
    tblt = get_word_table()
    path_dir = "validation.txt"
    canvas_x, assessment = import_corpus(path_dir, tblt)
    dataset = TensorDataset(
        torch.from_numpy(canvas_x).type(torch.float), torch.from_numpy(assessment).type(torch.long)
    )
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32, num_workers=2)