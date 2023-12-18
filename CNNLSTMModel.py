#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/13 20:06
# @Author  : liu
# @File    : CNNLSTMModel.py.py
# @Description : [1] Peng D, Li H, Dai Y, et al.
#                   Prediction of milling force based on spindle current signal by neural networks[J].
#                   Measurement, 2022, 205: 112153.


import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=80):
        super(CNNLSTMModel, self).__init__()

        # 定义CNN部分
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # 定义LSTM部分
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # 定义输出层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        # x = x.permute(0, 2, 1)

        # CNN部分
        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.conv2(x)
        # x = self.maxpool(x)

        # 将数据形状变换为LSTM的输入形状
        # x = x.permute(0, 2, 1)

        # LSTM部分
        LSTMout, _ = self.lstm(x)

        # 仅使用LSTM输出序列的最后一个时间步的结果
        LSTMout = LSTMout[:, -1, :]

        # 输出层
        out1 = self.fc1(LSTMout)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)

        return out3