import torch
from torch.utils.data import DataLoader
from data import *
import os
"""
先的到所有的数据
对数据进行处理 也就是读取文字，padding encoder 转换成 torch 定义 len getitem
"""
class Datasets:
    def __int__(self,config,data_path):
        self.config = config
        self.data_path = data_path

        if not os.path.isfile(self.config["vocab_path"]):
            vocab_build(self.config["vocab_path"],self.config['train_data_path'],self.config["min_count"])
        else:
            self.word2id = read_vocab(self.config['vocab_path'])
        self.input = read_corpus(data_path)
        self.inputIds = sentence2id(self.input,self.word2id)
        self.inputIds = pad_sequences(self.inputIds)
    def __len__(self):
        return len(self.inputIds)
    def __getitem__(self,index):
        return torch.LongTensor(self.inputIds[index])
def load_data(data_path, config, shuffle=True):
    datasets = Datasets(config, data_path)
    dataloader = DataLoader(datasets, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader