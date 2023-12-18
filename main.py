#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/10 15:29
# @Author  : Liu
# @File    : CustomDataset.py.py
# @Description : v0为了解决深度学习模型库的问题

import torch
import torch.nn as nn
import torch.nn.functional as F
from GeneralDataset import GeneralDataset
import pandas as pd

from AllModel import *

def initialize_model_parameters(model):
    """
    初始化模型参数
    """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def model_train(model, train_loader, optimizer, criterion, device):
    """
    模型训练函数
    """
    model.train()
    epoch_loss = 0

    for step, (X_i, Y_i) in enumerate(train_loader):
        X_i = X_i.to(device)
        Y_i = Y_i.to(device).float().unsqueeze(1)

        outputs_i = model(X_i)
        loss = criterion(outputs_i, Y_i)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        optimizer.zero_grad()

    return epoch_loss

def model_evaluate(model, valid_loader, device, DirTestResult):
    """
    模型评估函数
    """
    model.eval()
    valid_results = []

    with torch.no_grad():
        for step, (X_data, Y_data) in enumerate(valid_loader):
            X_data = X_data.to(device)
            Y_data = Y_data.to(device).float().unsqueeze(1)

            # 验证模型
            output_data = model(X_data)
            valid_results.append((output_data.item(), Y_data.item()))

    # 结果保存到 Excel 文件
    df = pd.DataFrame(valid_results, columns=["Predicted", "Actual"])
    df.to_excel(DirTestResult, index=False)
    print("Validation results saved to", DirTestResult)
    return 0

if __name__ == '__main__':
    '''参数设置'''
    num_epochs = 10  # 训练次数
    m = 20  # 特征数量
    dropout = 0  # 设置dropout rate

    '''地址与文件名设置'''
    DirTrainDf = './Dataset/4_6.xlsx'
    DirTestDf = './Dataset/phmall.xlsx'
    DirModel = './ModelParameters/4_6v1.pt'
    DirTestResult = './Result/4_6v1.xlsx'

    '''数据集读取'''
    train_df = pd.read_excel(DirTrainDf)  # 训练集
    train_dataset = GeneralDataset(train_df.iloc[:, :m], train_df.iloc[:, m])
    test_df = pd.read_excel(DirTestDf)    # 测试集
    test_dataset = GeneralDataset(test_df.iloc[:, :m], test_df.iloc[:, m])

    batch_size = 10

    '''数据加载器设置'''
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,    # 是否随机
                                               num_workers=4,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True)

    '''模型定义和加载'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = Transformer(m, dropout, device, d_model=128, N=4, heads=4).to(device)  # 此次模型可替换
    # model = SimpleCNN(m, 1).to(device)
    # model = SimpleLSTM(20,16,1).to(device)
    # model = SimpleRNN(20, 16, 1).to(device)
    # model = CNNLSTMModel(m).to(device)
    model = ResNetModel(input_size=m, hidden_size=64, output_size=1).to(device)

    '''初始化模型参数'''
    initialize_model_parameters(model)

    '''初始化优化器'''
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)    # Adagrad 可替换为Adam,AdamW,SGD等等

    '''损失函数'''
    criterion = torch.nn.MSELoss()

    '''模型训练'''
    for epoch in range(num_epochs):
        epoch_loss = model_train(model, train_loader, optimizer, criterion, device)
        print("Epoch: %d, training loss: %1.5f" % (epoch, epoch_loss))

    '''保存模型'''
    torch.save(model.state_dict(), DirModel)

    ###################################################################################
    '''模型加载'''
    model_state_dict = torch.load(DirModel)
    # model = Transformer(m, dropout, device, d_model=128 , N=4, heads=4)
    # model = SimpleCNN(m, 1).to(device)
    # model = SimpleLSTM(20, 16, 1).to(device)
    # model = SimpleRNN(20, 16, 1).to(device)
    # model = CNNLSTMModel(m).to(device)

    model.load_state_dict(model_state_dict)
    model.to(device)

    '''模型验证和测试结果保存'''
    valid_results = model_evaluate(model, valid_loader, device, DirTestResult)