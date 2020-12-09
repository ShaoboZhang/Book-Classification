# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Book Classification
@FileName: model.py
@Discribe: 
"""

import torch
import torch.nn as nn
import torch.utils.data as Data

from collections import Counter
from transformers import BertTokenizer, BertModel
from model.data_utils import preprocess

criterion = nn.CrossEntropyLoss(reduction='sum')


class BertDataset(Data.Dataset):
    def __init__(self, args, file_path, with_labels=True):
        self.data = preprocess(file_path, args.is_preprocess, with_labels)
        self.with_labels = with_labels
        self.tokenizer = BertTokenizer.from_pretrained(args.lib_path)
        self.max_len = args.max_len

    def __getitem__(self, item):
        sent = self.data.loc[item, 'text']
        encoded = self.tokenizer(text=sent, padding='max_length', truncation=True,
                                 max_length=self.max_len, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze()
        token_type_ids = encoded['token_type_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        if self.with_labels:
            labels = self.data.loc[item, 'category_id']
            return input_ids, token_type_ids, attention_mask, labels
        else:
            return input_ids, token_type_ids, attention_mask

    def __len__(self):
        return len(self.data)


class Model_Bert(nn.Module):
    def __init__(self, args):
        super(Model_Bert, self).__init__()
        hidden_sz = 256
        self.bert = BertModel.from_pretrained(args.lib_path)
        self.rnn = nn.GRU(args.hidden_size, hidden_sz, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(hidden_sz, args.label_num)
        )
        self.label_num = args.label_num
        self.device = args.device
        self.ignore_idx = torch.tensor(criterion.ignore_index).to(args.device)

    def forward(self, batch_data):
        input_ids, token_type_ids, attention_mask, labels = [data.to(self.device) for data in batch_data]
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        outputs, _ = self.rnn(outputs[0])
        logits = self.fc(torch.mean(outputs, dim=1))
        loss = criterion(logits, labels)
        acc_sum = accuracy(logits, labels)
        return loss, acc_sum


class BasicDataset(Data.Dataset):
    def __init__(self, args, file_path, with_labels=True, word2idx=None):
        self.data = preprocess(file_path, args.is_preprocess, with_labels)
        self.with_labels = with_labels
        self.max_len = args.max_len
        self.vocab_size = args.vocab_size
        self.word2idx = word2idx if word2idx else self._build_vocab(self.data['text'].values)

    def __getitem__(self, item):
        sent = self.data.loc[item, 'text']
        words = self._sent2idx(sent)
        if self.with_labels:
            label = self.data.loc[item, 'category_id']
            return words, label
        else:
            return words

    def __len__(self):
        return len(self.data)

    def _build_vocab(self, data):
        vocab = Counter()
        for sent in data:
            for word in sent.split():
                vocab.update(word)
        word2idx = {word: idx + 2 for idx, (word, _) in enumerate(vocab.most_common(self.vocab_size - 2))}
        word2idx['<pad>'] = 0
        word2idx['<unk>'] = 1
        return word2idx

    def _sent2idx(self, sent):
        words = [self.word2idx.get(word, 1) for word in sent if word!=' '][:self.max_len]
        words += (self.max_len - len(words)) * [0]
        return torch.tensor(words)


class Model_Basic(nn.Module):
    def __init__(self, args):
        super(Model_Basic, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.embed_size)
        self.rnn = nn.GRU(args.embed_size, args.hidden_size // 2, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_size, args.label_num)
        )
        self.device = args.device

    def forward(self, batch_data):
        sents, labels = [data.to(self.device) for data in batch_data]
        embed = self.embed(sents)
        _, hidden = self.rnn(embed)
        rnn_output = torch.cat([hidden[0], hidden[1]], dim=1)
        logits = self.classifier(rnn_output)
        loss = criterion(logits, labels)
        acc_sum = accuracy(logits, labels)
        return loss, acc_sum


def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    correct = torch.sum(torch.eq(preds, targets))
    return correct.item()
