#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 13:48
# @Author  : 我的名字
# @File    : TransformerModel.py.py
# @Description : 这个函数是用来balabalabala自己写

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from condconv import CondConv2D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
num_epochs = 1 # Number of training epochs
d_model = 128  # dimension in encoder
heads = 4  # number of heads in multi-head attention
N = 4  # number of encoder layers
m = 25  # number of features
'''

class Transformer(nn.Module):
    def __init__(self, m, dropout, device, d_model=128, N=4, heads=4):
        super().__init__()
        self.device = device
        self.gating = Gating(d_model, m).to(self.device)
        self.encoder = Encoder(d_model, N, heads, m, dropout).to(self.device)
        self.out1 = nn.Linear(d_model, 1).to(self.device)
        self.out2 = nn.Linear(16, 1).to(self.device)

        # self.w = nn.Parameter(torch.tensor(0.0, requires_grad=True)).to(device)
        # nn.Parameter(torch.randn(1, requires_grad=True)).to(device)

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1)  # Adding the convolution layer

    def forward(self, src):

        e_i = self.gating(src).to(device)
        e_outputs = self.encoder(e_i).to(device)
        output = self.out1(e_outputs).to(device)


        return output.reshape(1)



class Gating(nn.Module):
    def __init__(self, d_model, m): # 128,14
        super().__init__()
        self.m = m

        # the output
        self.W_e = nn.Parameter(torch.Tensor(m, d_model))
        self.b_e = nn.Parameter(torch.Tensor(d_model))

        self.init_weights()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1),
        )

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.m)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):

        x = x.reshape(1,self.m)
        return torch.matmul(x, self.W_e) + self.b_e # (the final output is 1,1,1,128 as the encoder has size of 128.)


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, m, dropout): #d_model = 128  # dimension in encoder, heads = 4  #number of heads in multi-head attention, N = 2  #encoder layers, m = 14  #number of features
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.d_model = d_model

    def forward(self, src):
        src = src.reshape(1, self.d_model) # this 128 is changed according to d_model
        # x = self.pe(src, t)
        for i in range(self.N):
            x = self.layers[i](src, None)
        return self.norm(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, t):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        pe = np.zeros(self.d_model)

        for i in range(0, self.d_model, 2):
            pe[i] = math.sin(t / (10000 ** ((2 * i) / self.d_model)))
            pe[i + 1] = math.cos(t / (10000 ** ((2 * (i-1)) / self.d_model)))

        x = x + Variable(torch.Tensor(pe)).to(device)
        return x


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.5):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.5):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
    # scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.5):
        super().__init__()
        # set d_ff as a default to 512
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.condconv = CondConv2D(d_ff, d_model, kernel_size=1, num_experts=3, dropout_rate=dropout)
        # self.condconv = nn.Conv2d(d_ff, d_model, kernel_size=1)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)

        # 使用permute方法交换维度
        # x_permuted = x.permute(2, 1, 0)
        # x_permuted = torch.unsqueeze(x_permuted, dim=0)
        # x = self.condconv(x_permuted)
        # x = torch.squeeze(x, dim=0)
        # x = x.permute(2, 1, 0)
        return x
