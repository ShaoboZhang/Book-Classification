# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Book Classification
@FileName: main.py
@Discribe: 
"""

from time import time
import torch.utils.data as Data

from model import config
from model.model import BertDataset, Model_Bert, BasicDataset, Model_Basic
from model.utils import train_eval, predict


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        func(*args, **kwargs)
        t1 = time()
        print(f"Processing time: {(t1 - t0) / 60:.2f}min")

    return wrapper


@timeit
def train():
    if config.use_bert:
        train_dataset = BertDataset(config, config.train_file_path)
        dev_dataset = BertDataset(config, config.dev_file_path)
        model = Model_Bert(config).to(config.device)
    else:
        train_dataset = BasicDataset(config, config.train_file_path)
        dev_dataset = BasicDataset(config, config.dev_file_path, word2idx=train_dataset.word2idx)
        model = Model_Basic(config).to(config.device)
    # 获取训练集
    train_data = Data.DataLoader(train_dataset, config.batch_size, shuffle=True)
    # 获取验证集
    dev_data = Data.DataLoader(dev_dataset, config.batch_size // 2)
    # 开始训练
    train_eval(model, train_data, dev_data, config)


def evaluate():
    if config.use_bert:
        test_dataset = BertDataset(config, config.test_file_path)
        model = Model_Bert(config).to(config.device)
    else:
        train_dataset = BasicDataset(config, config.train_file_path)
        test_dataset = BasicDataset(config, config.test_file_path, word2idx=train_dataset.word2idx)
        model = Model_Basic(config).to(config.device)
    # 获取测试集
    test_data = Data.DataLoader(test_dataset, config.batch_size)
    # 开始预测
    predict(model, test_data, config)


if __name__ == '__main__':
    train()
    evaluate()
