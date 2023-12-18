#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/16 15:43
# @Author  : 我的名字
# @File    : ResNetModel.py.py
# @Description : 这个函数是用来balabalabala自己写
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResNetModel, self).__init__()

        # 第一个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = Norm(hidden_size)

        # 第二个全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = Norm(hidden_size)

        # 输出层
        self.fc3 = nn.Linear(hidden_size, output_size)

        # 如果输入和输出的维度不同，需要调整维度
        self.shortcut = nn.Sequential()
        if input_size != output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, hidden_size),  # Corrected this line
                Norm(hidden_size)
            )

    def forward(self, x):
        residual = self.shortcut(x.view(x.size(0),-1) )  # 保存输入作为残差连接
        x = x.view(x.size(0),-1)  # Corrected this line
        out = self.fc1(x)

        out = F.relu(self.bn1(self.fc1(x)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = out + residual  # 残差连接
        out = F.relu(out)

        out = self.fc3(out)
        return out

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (
                    x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm