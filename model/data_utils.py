# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Book Classification
@FileName: data_utils.py
@Discribe: 
"""

import re
import pandas as pd


def preprocess(file_path, is_preprocess=True, with_labels=True):
    data = pd.read_csv(file_path, sep='\t')
    if is_preprocess:
        data['text'] = data['text'].apply(remove_punctuation)
        stopwords = get_stopwords()
        data['text'] = data['text'].apply(
            lambda x: " ".join([w.strip() for w in x.split(' ') if w not in stopwords]))
    if with_labels:
        return data[['text', 'category_id']]
    else:
        return data['text']


# 除去标点符号
def remove_punctuation(line):
    if not line: return
    line = str(line)
    rule = re.compile(u"[^A-Za-z0-9\u4300-\u9FA5]")
    line = rule.sub('', line)
    return line


# 获取停用词
def get_stopwords():
    stopwords = [line.strip() for line in open('./data/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
    return stopwords + ['']
