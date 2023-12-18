#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 14:25
# @Author  : 我的名字
# @File    : CNNModel.py.py
# @Description : 这个函数是用来balabalabala自己写
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16 * (input_size - 2), output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x