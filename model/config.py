# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Book Classification
@FileName: config.py
@Discribe: 
"""

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 数据文件路径
train_file_path = './data/train.csv'
dev_file_path = './data/dev.csv'
test_file_path = './data/test.csv'
stopwords_file_path = '.data/cn_stopwords.txt'

# 数据处理
is_preprocess = True
use_bert = False
lib_path = './lib/bert/'
vocab_size = 30000
max_len = 240

# 模型参数
batch_size = 64
embed_size = 300
hidden_size = 768 if use_bert else 256
dropout = 0.4
label_num = 33

# 训练参数
epoch_num = 4 if use_bert else 10
lr = 5e-5 if use_bert else 1e-4
weight_decay = 0.01

# 保存路径
if is_preprocess:
    model_path = f'./save/model_bert_with_prep.pt' if use_bert else f'./save/model_rnn_with_prep.pt'
else:
    model_path = f'./save/model_bert_without_prep.pt' if use_bert else f'./save/model_rnn_without_prep.pt'



