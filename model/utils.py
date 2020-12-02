# -*- coding: utf-8 -*-
"""
@Author  : Shaobo Zhang
@Project : Book Classification
@FileName: utils.py
@Discribe: 
"""

import os, torch
from time import sleep
from tqdm import tqdm
# from transformers import AdamW


from torch.optim import Adam

def train_eval(model, train_data, dev_data, args):
    def evaluate():
        torch.cuda.empty_cache()
        eval_loss = eval_acc = 0
        data_size = 0
        model.eval()
        with torch.no_grad():
            for batch_data in dev_data:
                loss, acc_sum = model(batch_data)
                eval_loss += loss.item()
                eval_acc += acc_sum
                data_size += len(batch_data[-1])
        model.train()
        return eval_loss / data_size, eval_acc / data_size

    # 模型保存目录若不存在，则创建目录
    dir_name = os.path.dirname(args.model_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # 开始训练
    print("开始训练...")
    optim = Adam(model.parameters(), lr=args.lr)
    model.train()
    sleep(0.5)
    eval_loss = best_loss = float('inf')
    eval_acc = 0.0
    for epoch in range(args.epoch_num):
        torch.cuda.empty_cache()
        with tqdm(train_data) as progress:
            progress.set_description(f'Epoch {epoch + 1}')
            for idx, batch_data in enumerate(train_data):
                optim.zero_grad()
                loss, acc_sum = model(batch_data)
                loss.backward()
                optim.step()
                # 若干轮之后，评估模型
                if (idx + 1) % 600 == 0:
                    eval_loss, eval_acc = evaluate()
                    # 若模型有提升，则保存模型
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        torch.save(model.state_dict(), args.model_path)
                batch_size = len(batch_data[-1])
                train_loss, train_acc = loss.item() / batch_size, acc_sum / batch_size
                progress.set_postfix(train_loss=train_loss, train_acc=train_acc, eval_loss=eval_loss, eval_acc=eval_acc)
                progress.update()


def predict(model, data, args):
    model.load_state_dict(torch.load(args.model_path))
    torch.cuda.empty_cache()
    eval_loss = eval_acc = 0
    data_size = 0
    model.eval()
    print("开始预测...")
    sleep(0.5)
    with torch.no_grad():
        for batch_data in tqdm(data):
            loss, acc_sum = model(batch_data)
            eval_loss += loss.item()
            eval_acc += acc_sum
            data_size += len(batch_data[-1])
    eval_loss /= data_size
    eval_acc /= data_size
    print(f'Eval Loss: {eval_loss:.3f}, Eval Accuracy: {eval_acc:.1%}')