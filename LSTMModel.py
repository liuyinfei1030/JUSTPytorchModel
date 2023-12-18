#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 14:30
# @Author  : 我的名字
# @File    : LSTMModel.py.py
# @Description : 这个函数是用来balabalabala自己写
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义简单的LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn.squeeze(0))
        return x