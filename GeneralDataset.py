#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 13:46
# @Author  : 我的名字
# @File    : GeneralDataset.py.py
# @Description : 这个函数是用来balabalabala自己写
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/22 14:10
# @Author  : 我的名字
# @File    : PHM2010Dataset.py.py
# @Description : PHM2010数据集

import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import re  # 导入正则表达式模块
import pywt              #Python中的小波分析库

class PHM2010_Dataset(Dataset):
    def __init__(self, transform=None):
        data_path = './data/PHM2010/'
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.datasets = []
        self.size_row = 50000

        def read_data_from_folder(folder_path):
            # 获取文件夹中的所有文件
            file_names = os.listdir(folder_path)
            # 过滤出CSV文件
            # 使用正则表达式提取数字部分并将其转换为整数作为排序键
            numeric_part = lambda x: int(re.findall(r'\d+', x)[0])
            csv_files = sorted([f for f in file_names if f.endswith('.csv')], key=numeric_part)
            return csv_files

        def signal_stats_all(data):
            # 24个特征公式：
            # time_domain:1-12
            # 绝对均值，特征1
            absolute_mean_value = np.sum(np.fabs(data)) / self.size_row
            # 峰值，特征2
            max = np.max(data)
            # 均方根值，特征3
            root_mean_score = np.sqrt(np.sum(np.square(data)) / self.size_row)
            # 方根幅值，特征4
            Root_amplitude = np.square(np.sum(np.sqrt(np.fabs(data))) / self.size_row)
            # 歪度值，特征5
            skewness = np.sum(np.power((np.fabs(data) - absolute_mean_value), 3)) / self.size_row
            # 峭度值，特征6
            Kurtosis_value = np.sum(np.power(data, 4)) / self.size_row
            # 波形因子，特征7
            shape_factor = root_mean_score / absolute_mean_value
            # 脉冲因子，特征8
            pulse_factor = max / absolute_mean_value
            # 歪度因子，特征9
            skewness_factor = skewness / np.power(root_mean_score, 3)
            # 峰值因子，特征10
            crest_factor = max / root_mean_score
            # 裕度因子，特征11
            clearance_factor = max / Root_amplitude
            # 峭度因子，特征12
            Kurtosis_factor = Kurtosis_value / np.power(root_mean_score, 4)

            # frequency_domain:13-16
            data_fft = np.fft.fft(data)  # fft,得到一系列的复数值，如a+bi（实部和虚部）:指实现用很多正弦波取表示这个信号
            Y = np.abs(data_fft)  # 对复数取绝对值=为该复数的模
            freq = np.fft.fftfreq(self.size_row, 1 / 50000)  # 获得频率；第一个参数是FFT的点数，一般取FFT之后的数据的长度（size）；
            # 第二个参数d是采样周期，其倒数就是采样频率Fs，即d=1/Fs，针对回转支承，采用频率为20KHZ
            ps = Y ** 2 / self.size_row  # 频域特性：指频率值及其对应的幅值
            # 重心频率，特征13
            FC = np.sum(freq * ps) / np.sum(ps)
            # 均方频率，特征14
            MSF = np.sum(ps * np.square(freq)) / np.sum(ps)
            # 均方根频率，特征15
            RMSF = np.sqrt(MSF)
            # 频率方差，特征16
            VF = np.sum(np.square(freq - FC) * ps) / np.sum(ps)

            # time and frequency domain:17-24
            wp = pywt.WaveletPacket(data, wavelet='db3', mode='symmetric', maxlevel=3)  # 小波包变换
            aaa = wp['aaa'].data
            aad = wp['aad'].data
            ada = wp['ada'].data
            add = wp['add'].data
            daa = wp['daa'].data
            dad = wp['dad'].data
            dda = wp['dda'].data
            ddd = wp['ddd'].data
            ret1 = np.linalg.norm(aaa, ord=None)  # 第一个节点系数求得的范数/ 矩阵元素平方和开方
            ret2 = np.linalg.norm(aad, ord=None)  # ord=None：默认情况下，是求整体的矩阵元素平方和，再开根号
            ret3 = np.linalg.norm(ada, ord=None)
            ret4 = np.linalg.norm(add, ord=None)
            ret5 = np.linalg.norm(daa, ord=None)
            ret6 = np.linalg.norm(dad, ord=None)
            ret7 = np.linalg.norm(dda, ord=None)
            ret8 = np.linalg.norm(ddd, ord=None)

            return [absolute_mean_value,
                max,
                root_mean_score,
                Root_amplitude,
                skewness,
                Kurtosis_value,
                shape_factor,
                pulse_factor,
                skewness_factor,
                crest_factor,
                clearance_factor,
                Kurtosis_factor,
                FC,
                MSF,
                RMSF,
                VF,
                # ret1,
                # ret2,
                # ret3,
                # ret4,
                # ret5,
                # ret6,
                # ret7,
                # ret8,
            ]

        def preprocess_data(data):
            sample_rate = 12
            resampled_data = data.iloc[::sample_rate, :7]
            standardized_data = stats.zscore(resampled_data)
            F_x_img = np.reshape(signal_stats_all(data.iloc[:, 0]), (4, 4))
            F_y_img = np.reshape(signal_stats_all(data.iloc[:, 1]), (4, 4))
            F_z_img = np.reshape(signal_stats_all(data.iloc[:, 2]), (4, 4))
            V_x_img = np.reshape(signal_stats_all(data.iloc[:, 3]), (4, 4))
            V_y_img = np.reshape(signal_stats_all(data.iloc[:, 4]), (4, 4))
            V_z_img = np.reshape(signal_stats_all(data.iloc[:, 5]), (4, 4))
            AE = np.reshape(signal_stats_all(data.iloc[:, 6]), (4, 4))


            # F_x = standardized_data.iloc[:, 0].reset_index(drop=True)
            # F_y = standardized_data.iloc[:, 1].reset_index(drop=True)
            # F_z = standardized_data.iloc[:, 2].reset_index(drop=True)
            # V_x = standardized_data.iloc[:, 3].reset_index(drop=True)
            # V_y = standardized_data.iloc[:, 4].reset_index(drop=True)
            # V_z = standardized_data.iloc[:, 5].reset_index(drop=True)
            # AE = standardized_data.iloc[:, 6].reset_index(drop=True)

            num_samples = 2500
            # 将垂直振动信号和水平振动信号分别reshape为50x50的2D图像，并将它们堆叠在一起形成2通道图像
            # F_x_img = np.reshape(F_x[:num_samples].values, (50, 50))
            # F_y_img = np.reshape(F_y[:num_samples].values, (50, 50))
            # F_z_img = np.reshape(F_x[:num_samples].values, (50, 50))
            # V_x_img = np.reshape(F_y[:num_samples].values, (50, 50))
            # V_y_img = np.reshape(F_x[:num_samples].values, (50, 50))
            # V_z_img = np.reshape(F_y[:num_samples].values, (50, 50))
            # AE = np.reshape(F_y[:num_samples].values, (50, 50))

            img = np.stack([F_x_img, F_y_img, F_z_img, V_x_img, V_y_img, V_z_img, AE], axis=0)

            img = torch.tensor(img, dtype=torch.float32)
            return img

        PHM2010_Data = [
            ("c1", read_data_from_folder(data_path + "c1")),
            # ("c2", read_data_from_folder(data_path + "c2")),
            # ("c3", read_data_from_folder(data_path + "c3")),
            ("c4", read_data_from_folder(data_path + "c4")),
            # ("c5", read_data_from_folder(data_path + "c5")),
            ("c6", read_data_from_folder(data_path + "c6")),
        ]

        for folder, files in PHM2010_Data:
            folder_path = os.path.join(self.data_path, folder)
            # processed_data.append(read_csv_data(folder_path, files))

            for idx, file in enumerate(files):
                file_path = os.path.join(folder_path, file)
                num_files = len(files)
                label = num_files - idx - 1  # 使用文件名作为标签
                # print(file)
                data = preprocess_data(pd.read_csv(file_path,header=None))
                self.datasets.append((data,label))
            # pd.DataFrame(self.datasets).to_excel('PHM2010.xlsx')

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        data, label = self.datasets[index]
        return data, label

class GeneralDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        data = np.reshape(self.features[index][0:20],(1,20))
        # data = self.features[index]
        label = self.labels[index]
        return data, label

